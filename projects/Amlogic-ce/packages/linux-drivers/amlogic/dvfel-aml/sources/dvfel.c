// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 djnice <djnice@freemail.hu>
/*
 * dvfel - Dolby Vision FEL composition node for Amlogic S5 (S928X)
 *
 * vfm node inserted between the HEVC decoder and the display path:
 *
 *     dvbldec (DV) / vdec.h265.00 -> dvfel -> amlvideo -> deinterlace -> amvideo
 *
 * For every 10-bit compressed (AFBC) decoder frame:
 *   (A) VICP decompresses it into a linear 10-bit 4:4:4 buffer,
 *   (G) a userspace compositor (/dev/dvfel, GPU) composes the enhancement
 *       layer into a second buffer; without a compositor the frame is kept,
 *   (B) VICP compresses the result back into an AFBC buffer owned by dvfel,
 *       or output12 publishes packed linear 12-bit 4:2:2 without VICP,
 * and a copy of the decoder vframe pointing at that buffer is passed on.
 * The copy keeps vf->index, so the Dolby Vision driver still fetches the
 * (already P8.1-converted) RPU from the decoder by index. The original
 * vframe is held until the display returns the copy.
 *
 * Two kernel threads form a pipeline so VICP and GPU work overlap:
 *   stage A: take decoder frames, (A), post GPU jobs      -> pending queue
 *   stage B: in display order wait for the job, (B), hand to display
 *
 * Linear layout: see dvfel_uapi.h. Unsupported frames (8-bit, not
 * compressed, too large, interlaced, mode 0) are passed through unchanged.
 * Supported frames wait for free buffers instead, so composed and
 * non-composed frames are never mixed because of buffer shortage.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/wait.h>
#include <linux/spinlock.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/debugfs.h>
#include <linux/uaccess.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/dma-buf.h>
#include <linux/dma-buf-map.h>
#include <linux/dma-resv.h>
#include <linux/dma-mapping.h>
#include <linux/scatterlist.h>
#include <linux/refcount.h>
#include <linux/fdtable.h>
#include <linux/file.h>
#include <linux/mm.h>
#include <linux/amlogic/media/vfm/vframe.h>
#include <linux/amlogic/media/vfm/vframe_provider.h>
#include <linux/amlogic/media/vfm/vframe_receiver.h>
#include <linux/amlogic/media/codec_mm/codec_mm.h>
#include <linux/amlogic/media/video_sink/video_keeper.h>
#include <linux/amlogic/media/vicp/vicp.h>
#include <linux/amlogic/media/ge2d/ge2d.h>
#include <linux/amlogic/media/amdolbyvision/dolby_vision.h>
#include <linux/amlogic/media/vout/vout_notify.h>
#include <linux/amlogic/media/registers/cpu_version.h>
#include <linux/amlogic/media/registers/register_map.h>
/* Use this kernel build's complete vendor type, never a copied ABI prefix. */
#include <drivers/media/vout/hdmitx21/hdmi_tx_module.h>

#include "dvfel_uapi.h"

/* Exported by amdv.c, but absent from this vendor public header. */
extern void set_amdv_enable(bool enable);

#define DRV_NAME	"dvfel"
/* receiver of the hardware decoded EL (vfm: "dveldec dvfelel") */
#define EL_RECV_NAME	"dvfelel"
#define EL_HOLD		8	/* EL pictures waiting for their BL frame */
#define EL_STALE	16	/* access units */
#define MAX_SLOTS	12
#define QUEUE_LEN	16
#define MAX_W		3840
#define MAX_H		2160

#define LIN_STRIDE(w)		((w) * 4)
#define LIN_SIZE(w, h)		DVFEL_BUF_SIZE(w, h)
#define LINEAR12_STRIDE(w)	ALIGN((w) * 3, 64)
#define LINEAR12_SIZE(w, h)	(LINEAR12_STRIDE(w) * (h))
/* AFBC output, 10-bit 4:2:0 worst case plus header/table */
#define FBC_BODY_SIZE(w, h)	PAGE_ALIGN((w) * (h) * 15 / 8 + 1024 * 1658 * 2)
#define FBC_HEAD_SIZE(w, h)	PAGE_ALIGN(ALIGN(w, 64) * ALIGN(h, 64) / 32)
#define FBC_TABLE_SIZE(w, h)	PAGE_ALIGN(FBC_BODY_SIZE(w, h) / PAGE_SIZE * 4)

static int mode = 2;
module_param(mode, int, 0644);
MODULE_PARM_DESC(mode, "0: passthrough, 1: VICP round trip, 2: GPU compositor when connected");

static int slots = 8;
module_param(slots, int, 0444);
MODULE_PARM_DESC(slots, "number of output AFBC buffers (2..12)");

static unsigned long long capture_pts;
module_param(capture_pts, ullong, 0644);
MODULE_PARM_DESC(capture_pts, "debug: capture the frame with this pts_us64 (0: off)");

static int gpu_timeout_ms = 150;
module_param(gpu_timeout_ms, int, 0644);
MODULE_PARM_DESC(gpu_timeout_ms, "max wait for the compositor before showing the base layer");

static int debug;
module_param(debug, int, 0644);

/*
 * Experiment: can the display take 12-bit 4:2:2 linear frames (24 bits per
 * pixel, the "old mode" 422 of the VD MIF)? In mode 1 every frame is
 * replaced by a static pattern Y 0x5a7, Cb 0x8c3, Cr 0x3d5 (12 bit) that the
 * VPP probe can read back. 1: word = Y << 12 | C, 2: word = C << 12 | Y.
 * 3 varies only the low two luma bits across four bands. 4 and 5 fill
 * every component with 0x555 or 0xaaa (all bytes 0x55 / 0xaa), allowing
 * the probe to distinguish quantization from byte-order errors.
 * 6 fills RGB888 (0x35, 0xa7, 0xda); 7 varies R=0x34..0x37 in four bands.
 */
static int test422;
module_param(test422, int, 0644);
MODULE_PARM_DESC(test422, "debug mode 1: 12-bit 422 test (1: Y high, 2: Y low, 3: low-two-bit bands, 4: all 0x555, 5: all 0xaaa, 6: RGB constant, 7: RGB low-bit bands)");

static int el_wait_ms = 20;
module_param(el_wait_ms, int, 0644);
MODULE_PARM_DESC(el_wait_ms, "max wait for the hardware decoded EL picture of a frame");

/* Opt in only after the VD MIF and Dolby input have passed the 12-bit probe. */
static bool output12;
module_param(output12, bool, 0644);
MODULE_PARM_DESC(output12, "output composed linear 12-bit 422 through the native Dolby Vision path");

/* Software carrier transport; userspace freezes PQ and selects the raster. */
static bool tvled;
module_param(tvled, bool, 0644);
MODULE_PARM_DESC(tvled, "publish GPU-composed Dolby Vision RGB888 carrier bytes (requires passthrough VPP/HDMI setup)");

static bool tvled_ge2d;
module_param(tvled_ge2d, bool, 0644);
MODULE_PARM_DESC(tvled_ge2d, "experimental GE2D RGBA to BGR carrier copy for order 0; first-frame byte validation and CPU fallback (default off)");

static bool tvled_pool;
module_param(tvled_pool, bool, 0444);
MODULE_PARM_DESC(tvled_pool, "experimental direct GPU writes to exported driver-owned RGB24 display pool (default off); finish GPU access and close pool FDs before Stop/unload");

static int tvled_order;
static int tvled_order_set(const char *val, const struct kernel_param *kp)
{
	int order, ret;

	ret = kstrtoint(val, 0, &order);
	if (ret)
		return ret;
	if (order < 0 || order > 5)
		return -EINVAL;
	WRITE_ONCE(*(int *)kp->arg, order);
	return 0;
}

static const struct kernel_param_ops tvled_order_ops = {
	.set = tvled_order_set,
	.get = param_get_int,
};
module_param_cb(tvled_order, &tvled_order_ops, &tvled_order, 0644);
MODULE_PARM_DESC(tvled_order, "GPU RGB to carrier memory order: 0=BGR, 1=RBG, 2=RGB, 3=GRB, 4=GBR, 5=BRG (test422 patterns unchanged)");

static bool tvled_vsif = true;
static int tvled_vsif_set(const char *val, const struct kernel_param *kp);
static const struct kernel_param_ops tvled_vsif_ops = {
	.set = tvled_vsif_set,
	.get = param_get_bool,
};
module_param_cb(tvled_vsif, &tvled_vsif_ops, &tvled_vsif, 0644);
MODULE_PARM_DESC(tvled_vsif, "debug: send DV VSIF when claiming/publishing a carrier (false keeps raw RGB bypass without clearing existing packets)");

static unsigned int tvled_src_dm_version = 2;
static int tvled_src_dm_version_set(const char *val, const struct kernel_param *kp)
{
	unsigned int version;
	int ret;

	ret = kstrtouint(val, 0, &version);
	if (ret)
		return ret;
	if (version > 7)
		return -EINVAL;
	WRITE_ONCE(*(unsigned int *)kp->arg, version);
	return 0;
}

static const struct kernel_param_ops tvled_src_dm_version_ops = {
	.set = tvled_src_dm_version_set,
	.get = param_get_uint,
};
module_param_cb(tvled_src_dm_version, &tvled_src_dm_version_ops,
		&tvled_src_dm_version, 0644);
MODULE_PARM_DESC(tvled_src_dm_version, "TV-led VSIF source DM version (0..7, production default 2)");

static unsigned int tvled_comp_map_post = 1;
static int tvled_comp_map_post_set(const char *val, const struct kernel_param *kp)
{
	unsigned int map;
	int ret;

	ret = kstrtouint(val, 0, &map);
	if (ret)
		return ret;
	if (map > 5)
		return -EINVAL;
	WRITE_ONCE(*(unsigned int *)kp->arg, map);
	return 0;
}

static const struct kernel_param_ops tvled_comp_map_post_ops = {
	.set = tvled_comp_map_post_set,
	.get = param_get_uint,
};
module_param_cb(tvled_comp_map_post, &tvled_comp_map_post_ops,
		&tvled_comp_map_post, 0644);
MODULE_PARM_DESC(tvled_comp_map_post, "S5 HDMI post component map for a TV-led carrier (0..5, measured default 1)");

static bool tvled_signal;
static int tvled_signal_set(const char *val, const struct kernel_param *kp);
static const struct kernel_param_ops tvled_signal_ops = {
	.set = tvled_signal_set,
	.get = param_get_bool,
};
module_param_cb(tvled_signal, &tvled_signal_ops, &tvled_signal, 0644);
MODULE_PARM_DESC(tvled_signal, "claim/release TV-led RGB8 HDMI signalling and bypass the native Dolby core (also automatic for TV-led jobs)");

static bool tvled_rgb8_prepare;
static int tvled_rgb8_prepare_set(const char *val, const struct kernel_param *kp);
static const struct kernel_param_ops tvled_rgb8_prepare_ops = {
	.set = tvled_rgb8_prepare_set,
	.get = param_get_bool,
};
module_param_cb(tvled_rgb8_prepare, &tvled_rgb8_prepare_ops, &tvled_rgb8_prepare, 0644);
MODULE_PARM_DESC(tvled_rgb8_prepare, "controlled S5 TV-led probe: own software RGB8 format preference for the next normal DRM mode change; native Dolby and VSIF must be off (default off)");

#define dvfel_dbg(fmt, ...) \
	do { if (debug) pr_info(DRV_NAME ": " fmt, ##__VA_ARGS__); } while (0)

/*
 * SLOT_HELD: taken by the display when the stream was reset. The display
 * may keep showing it (video keeper, e.g. across a decoder reset on seek),
 * so it is reused only once a frame of the new stream went out and the
 * keeper no longer holds the header buffer.
 */
enum slot_state { SLOT_FREE, SLOT_BUSY, SLOT_OUT, SLOT_HELD };

/* This allocation identity never owns a dma-buf. It holds one fixed shared
 * codec_mm reference for its whole lifetime: the slot has one metadata ref,
 * and each exporter adds another. Thus FD churn never changes the codec_mm
 * reference baseline while slot_kept examines it in interrupt context.
 */
struct dvfel_pool_mem {
	refcount_t refs;
	atomic_t exporters;
	struct codec_mm_s *mm;
	ulong phys;
	size_t size;
	u32 orphan_writer;
	bool abandoned;
};

struct dvfel_pool_export {
	struct dvfel_pool_mem *mem;
	struct mutex lock;
	struct list_head attachments;
};

struct dvfel_pool_attachment {
	struct list_head node;
	struct sg_table table;
	struct device *dev;
	bool mapped;
};

struct dvfel_slot {
	enum slot_state state;
	bool taken;		/* SLOT_OUT frame handed to the display */
	bool tvled_carrier;	/* published from a successful DONE_TVLED job */
	u32 gen;
	struct vframe_s *orig;
	struct vframe_s out_vf;
	/* the header handle lets the video keeper hold the frame */
	struct codec_mm_s *body_mm, *head_mm;
	bool body_tied;		/* body released by the header's release */
	ulong body_phys, head_phys, table_phys, table_handle;
	struct codec_mm_s *linear_mm;
	struct dvfel_pool_mem *pool_mem;
	u32 pool_writer;
	ulong linear_phys;	/* per-slot packed 12-bit 422; kept like AFBC memory */
	bool linear_ge2d_prepared; /* only probe lines have CPU access between device copies */
};

struct dvfel_pass {
	struct vframe_s *vf;
	u32 gen;
};

enum job_state {
	JOB_FREE,
	JOB_RESERVED,	/* stage A is decompressing into in[] */
	JOB_POSTED,	/* waiting for the compositor */
	JOB_TAKEN,	/* compositor is working on it */
	JOB_DONE,	/* compositor finished */
	JOB_FINISH,	/* stage B is compressing from in[]/out[] */
};

struct dvfel_kjob {
	enum job_state state;
	bool orphan;		/* stage B gave up; free when the compositor reports */
	bool capture;		/* debug capture at stage B */
	u32 id;
	u32 gen;
	u32 flags;
	u64 pool_token;
	int status;
	u64 pts_us;
	u32 bl_au, el_au;	/* access units of the pair (PAIR_AU) */
	ktime_t posted;
	struct dvfel_slot *slot;
	struct vframe_s *orig;
};

struct dvfel_cbuf {
	struct dma_buf *db;
	struct dma_buf_attachment *att;
	struct sg_table *sgt;
	struct dma_buf_map map;
	bool mapped;
	ulong phys;
};

/* pending queue entry, in display order */
struct dvfel_pend {
	struct vframe_s *vf;	/* ready frame (processed copy or passthrough) */
	int job;		/* >= 0: wait for this job instead */
	u32 gen;
};

/* VICP descriptors per thread (too big for the stack) */
struct vicp_ctx {
	struct vicp_data_config_s cfg;
	struct dma_data_config_s dma;
	struct vframe_s vin;
};

struct dvfel_dev {
	spinlock_t lock;
	struct vframe_receiver_s recv;
	struct vframe_provider_s prov;
	bool prov_reg;
	bool bl_reset_pending;
	u32 bl_users;
	wait_queue_head_t wq_bl_idle;
	u32 gen;
	int nslots;

	struct vframe_s *outq[QUEUE_LEN];
	int outq_head, outq_cnt;
	struct dvfel_pend pendq[QUEUE_LEN];
	int pend_head, pend_cnt;
	struct dvfel_slot slots[MAX_SLOTS];
	struct dvfel_pass pass[QUEUE_LEN];

	struct task_struct *thread_a, *thread_b;
	wait_queue_head_t wq_a, wq_b;
	bool kick_a, kick_b;
	bool new_stream;
	bool shown_new;		/* the display took a frame since the last reset */

	/* compositor (userspace) */
	struct miscdevice misc;
	struct mutex client_lock;
	struct file *client;
	bool client_ready;
	u32 client_w, client_h;
	int client_nbufs;
	u64 pool_token;
	bool pool_ready;
	u64 client_pool_token;
	struct dvfel_cbuf cin[DVFEL_MAX_BUFS], cout[DVFEL_MAX_BUFS];
	/* hardware decoded EL: receiver of the EL decoder, client EL buffers */
	struct vframe_receiver_s el_recv;
	bool el_prov;
	bool el_reset_pending;
	u32 el_users;
	wait_queue_head_t wq_el_idle;
	spinlock_t el_lock;
	struct vframe_s *el_hold[EL_HOLD];
	int el_nhold;
	u32 client_el_w, client_el_h;
	struct dvfel_cbuf cel[DVFEL_MAX_BUFS];
	struct dvfel_kjob jobs[DVFEL_MAX_BUFS];
	u32 job_seq;
	wait_queue_head_t wq_client;

	/* dvfel owned buffers, allocated on the first processable frame */
	struct mutex buf_lock;
	bool bufs_ok;
	ulong lin_phys;
	u32 buf_w, buf_h;
	struct delayed_work free_work;

	struct vicp_ctx ctx_a, ctx_b;
	struct ge2d_context_s *ge2d;
	bool ge2d_verified, ge2d_failed;
	u32 ge2d_copies;
	bool signal_owned, signal_saved_enable, signal_saved_native_on, vsif_on;
	bool tvled_started;	/* this generation delivered a signalled carrier */
	ktime_t tvled_first_delivery;	/* monotonic, valid while tvled_started */
	/* DISPLAY opt-in is independent of decoder generations and GPU clients. */
	struct file *display_owner;
	u64 display_session, display_epoch, display_sequence;
	u32 display_width, display_height;
	bool display_lifecycle, display_ready, display_prepared;
	bool display_saved_vsif, display_native_saved, display_saved_enable;
	u32 signal_saved_hdr_ctrl[4];
	u32 signal_saved_osd_bits[6];
	u32 signal_saved_fmt_bits, signal_saved_dith_bits;
	u32 signal_saved_post_map_bits;
	u32 signal_saved_dv_bits[6];
	bool signal_active_warned;
	u64 signal_active_rejects;
	bool stopping;

	/* stats */
	u64 frames_in, frames_proc, frames_pass, vicp_err, slot_waits, held_breaks;
	u64 gpu_jobs, gpu_composed, gpu_fallback, gpu_timeouts;
	u64 linear12_frames, linear12_fallback;
	u64 tvled_frames, tvled_dropped;
	u64 tvled_pool_frames;
	u64 tvled_delivered, tvled_unique_delivered, tvled_same_pts, tvled_last_pts;
	u32 tvled_last_gen;
	bool tvled_have_pts;
	u64 tvled_ge2d_frames, tvled_ge2d_fallback;
	u64 tvled_ge2d_golden_checks;
	u64 el_in, el_used, el_dropped, el_missing;
	s64 us_a_sum, us_b_sum, us_a_max, us_b_max, us_gpu_sum, us_gpu_max;
	s64 us_el_sum, us_el_max, us_el_wait_max;
	s64 us_tvled_sum, us_tvled_max;
	u64 us_cnt, us_b_cnt, us_gpu_cnt, us_el_cnt;
	u64 us_tvled_cnt;

	/* debug capture of the next frame (A output and A of our B output) */
	struct dentry *dbg;
	atomic_t capture_req;
	void *cap_buf;
	size_t cap_size;
	int cap_frames;
	u64 cap_pts_us;
	u32 cap_w, cap_h;
};

static struct dvfel_dev *gdev;

static void pool_mem_put(struct dvfel_pool_mem *p);

/* d->lock held. Zero is reserved; advance on reset/allocation replacement. */
static void pool_next_token_locked(struct dvfel_dev *d)
{
	if (!++d->pool_token)
		++d->pool_token;
}

/* Software callback/delivery state only; never access HDMI from these
 * helpers. The delivery milestone is not an acknowledgement from the TV.
 */
static void tvled_signal_state(struct dvfel_dev *d, bool active)
{
	unsigned long flags;

	if (!d)
		return;
	spin_lock_irqsave(&d->lock, flags);
	d->vsif_on = active && READ_ONCE(tvled_vsif) && READ_ONCE(d->signal_owned);
	if (!d->vsif_on) {
		d->tvled_started = false;
		d->tvled_first_delivery = 0;
	}
	spin_unlock_irqrestore(&d->lock, flags);
}

/* Keep startup scene refreshes alive while the sink enters DV. This is a
 * bounded startup allowance after software delivery, not a TV acknowledgement.
 */
#define TVLED_STARTUP_REFRESH_MS 2000

/* d->lock held. Old clients ignore this additional bit; on an old kernel
 * it is always absent, preserving their conservative startup behaviour.
 */
static u32 tvled_job_flags_locked(struct dvfel_dev *d)
{
	u32 flags = DVFEL_JOB_TVLED;

	if ((!d->display_lifecycle || d->display_ready) &&
	    d->tvled_started && d->vsif_on && READ_ONCE(tvled_vsif) &&
	    ktime_compare(ktime_get(), ktime_add_ms(d->tvled_first_delivery,
						 TVLED_STARTUP_REFRESH_MS)) >= 0)
		flags |= DVFEL_JOB_TVLED_STARTED;
	return flags;
}

/* Only successful GPU carrier output can establish the milestone. Raw RGB
 * test patterns, passthrough frames and old reset epochs cannot do so.
 * d->lock held, including when called from the display's vsync IRQ.
 */
static void tvled_delivered_locked(struct dvfel_dev *d, struct dvfel_slot *s,
				   struct vframe_s *vf)
{
	if ((!d->display_lifecycle || d->display_ready) &&
	    s && s->state == SLOT_OUT && s->gen == d->gen && s->tvled_carrier &&
	    d->prov_reg && !READ_ONCE(d->stopping) && d->vsif_on &&
	    READ_ONCE(d->signal_owned) && READ_ONCE(tvled_vsif) &&
	    (vf->type & (VIDTYPE_VIU_444 | VIDTYPE_RGB_444)) ==
			(VIDTYPE_VIU_444 | VIDTYPE_RGB_444) &&
	    s->linear_mm && vf->mem_handle == s->linear_mm) {
		if (!d->tvled_started) {
			d->tvled_first_delivery = ktime_get();
			d->tvled_started = true;
		}
		d->tvled_delivered++;
		if (!d->tvled_have_pts || d->tvled_last_gen != d->gen ||
		    d->tvled_last_pts != vf->pts_us64)
			d->tvled_unique_delivered++;
		else
			d->tvled_same_pts++;
		d->tvled_have_pts = true;
		d->tvled_last_gen = d->gen;
		d->tvled_last_pts = vf->pts_us64;
	}
}

/* vf_get() only pins the provider during its get operation, not while a
 * consumer owns the returned VF. Decoder UNREG may free its entire VF pool.
 * These gates cover every use of an upstream pointer, including VICP DMA.
 * Acquisition is IRQ safe; teardown waits only after closing the gate and
 * releasing all spinlocks. A/B remain alive until the final module drain.
 */
static bool bl_ref_get_locked(struct dvfel_dev *d)
{
	if (!d->prov_reg || READ_ONCE(d->stopping))
		return false;
	d->bl_users++;
	return true;
}

static void bl_ref_put(struct dvfel_dev *d)
{
	unsigned long flags;
	bool idle;

	spin_lock_irqsave(&d->lock, flags);
	if (WARN_ON_ONCE(!d->bl_users)) {
		spin_unlock_irqrestore(&d->lock, flags);
		return;
	}
	idle = !--d->bl_users;
	spin_unlock_irqrestore(&d->lock, flags);
	if (idle)
		wake_up_all(&d->wq_bl_idle);
}

static bool el_ref_get(struct dvfel_dev *d)
{
	unsigned long flags;
	bool open;

	spin_lock_irqsave(&d->el_lock, flags);
	open = d->el_prov && !READ_ONCE(d->stopping);
	if (open)
		d->el_users++;
	spin_unlock_irqrestore(&d->el_lock, flags);
	return open;
}

static void el_ref_put(struct dvfel_dev *d)
{
	unsigned long flags;
	bool idle;

	spin_lock_irqsave(&d->el_lock, flags);
	if (WARN_ON_ONCE(!d->el_users)) {
		spin_unlock_irqrestore(&d->el_lock, flags);
		return;
	}
	idle = !--d->el_users;
	spin_unlock_irqrestore(&d->el_lock, flags);
	if (idle)
		wake_up_all(&d->wq_el_idle);
}

/* Never take this mutex while holding buf_lock or the vframe spinlock: the
 * transmitter callback takes its own mode/EDID locks. Its state and native
 * Dolby ownership are restored as one transaction on stream/client stop.
 */
static DEFINE_MUTEX(tvled_signal_lock);
/* S5 only: one HDR controller for each possible VD1 processing slice. */
static const u32 s5_tvled_hdr_ctrl[] = { 0x25a0, 0x26a0, 0x2ca0, 0x3ea0 };
/* OSD HDR has separate input/output matrix gates even when its HDR core is off. */
static const struct {
	u32 reg, mask;
} s5_tvled_osd_bypass[] = {
	{ 0x6450, 0xffffffffU },
	{ 0x648b, BIT(0) },
	{ 0x648c, BIT(0) },
	{ 0x6c50, 0xffffffffU },
	{ 0x6c8b, BIT(0) },
	{ 0x6c8c, BIT(0) },
};
/* S5 amdv_regs_s5.h: Core3 top enable, four slice enables, then the two
 * Core2 OSD bypass bits. Leave clocks and every unowned field unchanged.
 */
static const struct {
	u32 reg, mask, bypass;
} s5_tvled_dv_bypass[] = {
	{ 0x10fd, BIT(11), 0 },
	{ 0x2501, BIT(3), 0 },
	{ 0x2601, BIT(3), 0 },
	{ 0x2c01, BIT(3), 0 },
	{ 0x3e01, BIT(3), 0 },
	{ 0x6077, BIT(0) | BIT(4), BIT(0) | BIT(4) },
};
/* S5 VCBUS addresses from hdmitx21/hw/register.h. These are VPU registers,
 * not the HDMI packet block that may be in reset during a mode setting.
 */
#define S5_TVLED_HDMI_FMT_CTRL	0x2743
#define S5_TVLED_HDMI_DITH_CNTL	0x27fc
#define S5_TVLED_HDMI_SETTING	0x271b
#define S5_TVLED_FMT_MASK		(BIT(0) | BIT(1) | BIT(4) | BIT(10) | BIT(11))
#define S5_TVLED_DITH_MASK	(BIT(4) | BIT(10))
#define S5_TVLED_POST_MAP_MASK	GENMASK(7, 5)

static void tvled_update_vcbus_bits(u32 reg, u32 mask, u32 bits)
{
	u32 value = aml_read_vcbus(reg);

	aml_write_vcbus(reg, (value & ~mask) | (bits & mask));
}

/* tvled_signal_lock held; READY additionally holds the actual TX mode lock. */
static bool tvled_signal_acquire_locked(struct dvfel_dev *d, u32 gen, bool manual)
{
	struct vinfo_s *v;
	struct dv_vsif_para para = { 0 };
	bool on = false, signal_checked = false, vsif_active = false;
	int i;

	if (!d || READ_ONCE(d->stopping) || !is_meson_s5_cpu() || (!manual &&
	    (!READ_ONCE(d->prov_reg) || !READ_ONCE(d->client_ready) ||
	     gen != READ_ONCE(d->gen))))
		goto out;
	signal_checked = true;
	if (!d->signal_owned) {
		bool active;

		d->signal_saved_enable = is_amdv_enable();
		d->signal_saved_native_on = is_amdv_on();
		active = d->signal_saved_native_on;
		for (i = 0; i < ARRAY_SIZE(s5_tvled_dv_bypass); i++) {
			d->signal_saved_dv_bits[i] =
				aml_read_vcbus(s5_tvled_dv_bypass[i].reg) &
				s5_tvled_dv_bypass[i].mask;
			if (d->signal_saved_dv_bits[i] != s5_tvled_dv_bypass[i].bypass)
				active = true;
		}
		/* Native BYPASS must complete through its own vsync path first.
		 * The exported bool setter cannot shut down/reset an active core;
		 * do not seize it or later revive stale active routing state.
		 */
		if (active) {
			d->signal_active_rejects++;
			if (!d->signal_active_warned) {
				pr_warn(DRV_NAME ": TVled claim rejected: native on=%d enable=%d, TOP11=%#x slice3=%#x/%#x/%#x/%#x OSD0/4=%#x; complete native BYPASS first\n",
					d->signal_saved_native_on, d->signal_saved_enable,
					d->signal_saved_dv_bits[0], d->signal_saved_dv_bits[1],
					d->signal_saved_dv_bits[2], d->signal_saved_dv_bits[3],
					d->signal_saved_dv_bits[4], d->signal_saved_dv_bits[5]);
				d->signal_active_warned = true;
			}
			goto out;
		}
		for (i = 0; i < ARRAY_SIZE(s5_tvled_hdr_ctrl); i++)
			d->signal_saved_hdr_ctrl[i] = aml_read_vcbus(s5_tvled_hdr_ctrl[i]);
		for (i = 0; i < ARRAY_SIZE(s5_tvled_osd_bypass); i++)
			d->signal_saved_osd_bits[i] =
				aml_read_vcbus(s5_tvled_osd_bypass[i].reg) &
				s5_tvled_osd_bypass[i].mask;
		d->signal_saved_fmt_bits = aml_read_vcbus(S5_TVLED_HDMI_FMT_CTRL) &
			S5_TVLED_FMT_MASK;
		d->signal_saved_dith_bits = aml_read_vcbus(S5_TVLED_HDMI_DITH_CNTL) &
			S5_TVLED_DITH_MASK;
		d->signal_saved_post_map_bits = aml_read_vcbus(S5_TVLED_HDMI_SETTING) &
			S5_TVLED_POST_MAP_MASK;
		d->signal_owned = true;
	}
	/* DRM/window recreation can enable the native core again. */
	set_amdv_enable(false);
	/* The public enable setter only changes a bool; it does not route
	 * around already enabled Core2/3 hardware. Bypass that routing here.
	 */
	for (i = 0; i < ARRAY_SIZE(s5_tvled_dv_bypass); i++)
		tvled_update_vcbus_bits(s5_tvled_dv_bypass[i].reg,
					s5_tvled_dv_bypass[i].mask,
					s5_tvled_dv_bypass[i].bypass);
	/* csc_en must already be frozen by userspace. Clearing only bit 13
	 * leaves matrix-only processing alive: input/output matrices and the
	 * only_mat path have independent enables in bits 14/15/16. A carrier
	 * bypasses every HDR sub-block; restore the saved full word on release.
	 */
	for (i = 0; i < ARRAY_SIZE(s5_tvled_hdr_ctrl); i++)
		aml_write_vcbus(s5_tvled_hdr_ctrl[i], 0);
	for (i = 0; i < ARRAY_SIZE(s5_tvled_osd_bypass); i++)
		tvled_update_vcbus_bits(s5_tvled_osd_bypass[i].reg,
					s5_tvled_osd_bypass[i].mask, 0);
	WRITE_ONCE(tvled_signal, true);
	if (READ_ONCE(tvled_vsif)) {
		v = get_current_vinfo();
		if (!v || !v->vout_device || !v->vout_device->fresh_tx_vsif_pkt)
			goto out;
		para.ver = 1;
		para.length = 0x1b;
		para.vers.ver2.low_latency = 0;
		para.vers.ver2.dobly_vision_signal = 1;
		para.vers.ver2.src_dm_version = READ_ONCE(tvled_src_dm_version);
		v->vout_device->fresh_tx_vsif_pkt(EOTF_T_DOLBYVISION, RGB_8BIT, &para, false);
		vsif_active = true;
	}
	/* The VSIF callback reconfigures dithering, including round-to-8-bit.
	 * Reassert carrier bypass after it, retaining the current mode's sync
	 * polarity and all other fields. Never replay a prior mode's full word.
	 */
	tvled_update_vcbus_bits(S5_TVLED_HDMI_FMT_CTRL, S5_TVLED_FMT_MASK, 0);
	tvled_update_vcbus_bits(S5_TVLED_HDMI_DITH_CNTL, S5_TVLED_DITH_MASK, 0);
	/* Keep current timing, pre-map and FRL fields across mode changes. */
	tvled_update_vcbus_bits(S5_TVLED_HDMI_SETTING, S5_TVLED_POST_MAP_MASK,
				READ_ONCE(tvled_comp_map_post) << 5);
	on = true;
 out:
	if (signal_checked)
		tvled_signal_state(d, vsif_active);
	return on;
}

static bool tvled_signal_acquire(struct dvfel_dev *d, u32 gen, bool manual)
{
	bool on;

	mutex_lock(&tvled_signal_lock);
	if (d && READ_ONCE(d->display_lifecycle))
		on = READ_ONCE(d->display_ready);
	else
		on = tvled_signal_acquire_locked(d, gen, manual);
	mutex_unlock(&tvled_signal_lock);
	return on;
}

static void tvled_signal_release(struct dvfel_dev *d)
{
	int i;

	mutex_lock(&tvled_signal_lock);
	/* Close delivery before restoring native state. A concurrent get cannot
	 * re-establish STARTED using the previous callback's active flag.
	 */
	tvled_signal_state(d, false);
	if (d && d->signal_owned) {
		/* Do not clear the HDMI packet here. A stream/client stop can
		 * overlap a GUI mode setting, and the transmitter permits NULL
		 * clears even with ready == 0. Such a write during HDMI reset can
		 * raise an asynchronous SError. The mode driver owns packet clear
		 * under its mode-setting lock; a ready-state read is insufficient.
		 */
		for (i = 0; i < ARRAY_SIZE(s5_tvled_hdr_ctrl); i++)
			aml_write_vcbus(s5_tvled_hdr_ctrl[i], d->signal_saved_hdr_ctrl[i]);
		for (i = ARRAY_SIZE(s5_tvled_osd_bypass) - 1; i >= 0; i--)
			tvled_update_vcbus_bits(s5_tvled_osd_bypass[i].reg,
						s5_tvled_osd_bypass[i].mask,
						d->signal_saved_osd_bits[i]);
		tvled_update_vcbus_bits(S5_TVLED_HDMI_FMT_CTRL, S5_TVLED_FMT_MASK,
					d->signal_saved_fmt_bits);
		tvled_update_vcbus_bits(S5_TVLED_HDMI_DITH_CNTL, S5_TVLED_DITH_MASK,
					d->signal_saved_dith_bits);
		tvled_update_vcbus_bits(S5_TVLED_HDMI_SETTING, S5_TVLED_POST_MAP_MASK,
					d->signal_saved_post_map_bits);
		/* Only inactive native routing can have passed the claim guard.
		 * Restore its owned fields without replaying any unrelated bits.
		 */
		for (i = ARRAY_SIZE(s5_tvled_dv_bypass) - 1; i >= 0; i--)
			tvled_update_vcbus_bits(s5_tvled_dv_bypass[i].reg,
						s5_tvled_dv_bypass[i].mask,
						d->signal_saved_dv_bits[i]);
		set_amdv_enable(d->signal_saved_enable);
		d->signal_owned = false;
	}
	WRITE_ONCE(tvled_signal, false);
	mutex_unlock(&tvled_signal_lock);
}

static int tvled_signal_set(const char *val, const struct kernel_param *kp)
{
	bool requested;
	struct kernel_param parsed = *kp;
	int ret;

	parsed.arg = &requested;
	ret = param_set_bool(val, &parsed);
	if (ret)
		return ret;
	/* Module arguments are parsed before gdev has been created. */
	if (!gdev) {
		WRITE_ONCE(tvled_signal, requested);
		return 0;
	}
	if (requested)
		tvled_signal_acquire(gdev, 0, true);
	else
		tvled_signal_release(gdev);
	return 0;
}

static int tvled_vsif_set(const char *val, const struct kernel_param *kp)
{
	bool requested;
	struct kernel_param parsed = *kp;
	int ret;

	parsed.arg = &requested;
	ret = param_set_bool(val, &parsed);
	if (ret)
		return ret;
	mutex_lock(&tvled_signal_lock);
	WRITE_ONCE(tvled_vsif, requested);
	if (!requested)
		tvled_signal_state(gdev, false);
	/* Enabling the debug gate does not claim signalling by itself. The
	 * next explicit claim/publish must invoke the callback first.
	 */
	mutex_unlock(&tvled_signal_lock);
	return 0;
}

/* Independent of signal ownership: a normal modeset can reset the provider.
 * Lock order is attr -> transmitter mode; never nest tvled_signal_lock here.
 * These helpers change only fmt_attr, never registers, VSIF or display modes.
 */
static DEFINE_MUTEX(tvled_attr_lock);
static bool tvled_attr_saved;
static char tvled_saved_fmt_attr[16];
static struct hdmitx_common *tvled_attr_tx;

/* Both locks held. hdmitx_setup_attr zeroes bytes after the first NUL; only
 * the preference string has semantics, while the original snapshot is 16B.
 */
static int tvled_attr_write_checked(struct hdmitx_common *tx, const char *attr)
{
	char actual[16];
	int ret;

	ret = hdmitx_setup_attr(tx, attr);
	if (ret)
		return ret;
	ret = hdmitx_get_attr(tx, actual);
	if (ret)
		return ret;
	if (!memchr(actual, '\0', sizeof(actual)) || strcmp(actual, attr))
		return -EIO;
	return 0;
}

static int tvled_attr_update(bool requested)
{
	struct hdmitx_dev *hdev;
	struct hdmitx_common *tx;
	int ret = 0, rollback;

	mutex_lock(&tvled_attr_lock);
	if (!requested && !tvled_attr_saved)
		goto done;
	if (requested) {
		if (!gdev || READ_ONCE(gdev->stopping)) {
			ret = -ENODEV;
			goto done;
		}
		if (!is_meson_s5_cpu() || !READ_ONCE(tvled)) {
			ret = -EOPNOTSUPP;
			goto done;
		}
		if (READ_ONCE(tvled_vsif)) {
			ret = -EBUSY;
			goto done;
		}
		hdev = get_hdmitx21_device();
		if (!hdev) {
			ret = -ENODEV;
			goto done;
		}
		tx = &hdev->tx_comm;
		if (tvled_attr_saved && tx != tvled_attr_tx) {
			ret = -ESTALE;
			goto done;
		}
	} else {
		tx = tvled_attr_tx;
	}
	mutex_lock(&tx->hdmimode_mutex);
	if (requested) {
		/* An enabled native path makes decide_color_attr force DV colors
		 * instead of reading fmt_attr. The existing owner must finish
		 * native BYPASS first; this bridge owns no native-enable state.
		 */
		if (is_amdv_on() || is_amdv_enable()) {
			ret = -EBUSY;
			goto unlock_tx;
		}
		if (!tvled_attr_saved) {
			ret = hdmitx_get_attr(tx, tvled_saved_fmt_attr);
			if (ret)
				goto unlock_tx;
			if (!memchr(tvled_saved_fmt_attr, '\0', sizeof(tvled_saved_fmt_attr))) {
				ret = -EINVAL;
				goto unlock_tx;
			}
			tvled_attr_tx = tx;
			tvled_attr_saved = true;
		}
		ret = tvled_attr_write_checked(tx, "rgb,8bit");
		if (!ret)
			goto unlock_tx;
		rollback = tvled_attr_write_checked(tx, tvled_saved_fmt_attr);
		if (rollback) {
			pr_warn(DRV_NAME ": RGB8 preference failed (%d), rollback failed (%d)\n",
				ret, rollback);
			goto unlock_tx; /* keep ownership for a later disable/exit */
		}
	} else {
		ret = tvled_attr_write_checked(tx, tvled_saved_fmt_attr);
		if (ret)
			goto unlock_tx;
	}
	tvled_attr_saved = false;
	tvled_attr_tx = NULL;
 unlock_tx:
	mutex_unlock(&tx->hdmimode_mutex);
 done:
	/* Remain owned if a failed write could not be rolled back. Callers
	 * must still check the setter's error before starting a modeset.
	 */
	WRITE_ONCE(tvled_rgb8_prepare, tvled_attr_saved);
	mutex_unlock(&tvled_attr_lock);
	return ret;
}

static int tvled_rgb8_prepare_set(const char *val, const struct kernel_param *kp)
{
	bool requested;
	struct kernel_param parsed = *kp;
	int ret;

	parsed.arg = &requested;
	ret = param_set_bool(val, &parsed);
	if (ret)
		return ret;
	/* No bridge work while module arguments are parsed, before gdev exists. */
	if (!gdev)
		return requested ? -EAGAIN : 0;
	return tvled_attr_update(requested);
}

/* DISPLAY -> signal -> TX mode. Attribute helpers use their own lock -> TX
 * mode separately, never nested with signal. No notifier callback takes any
 * of these mutexes. DQ remains enabled: Kodi needs it for first Configure.
 */
static DEFINE_MUTEX(tvled_display_lock);

static void tvled_display_quiesce(struct dvfel_dev *d)
{
	unsigned long flags;

	mutex_lock(&tvled_signal_lock);
	spin_lock_irqsave(&d->lock, flags);
	d->display_ready = false;
	d->display_prepared = false;
	spin_unlock_irqrestore(&d->lock, flags);
	tvled_signal_state(d, false);
	mutex_unlock(&tvled_signal_lock);
}

static bool tvled_native_inactive(void)
{
	unsigned int i;

	if (is_amdv_on())
		return false;
	for (i = 0; i < ARRAY_SIZE(s5_tvled_dv_bypass); i++)
		if ((aml_read_vcbus(s5_tvled_dv_bypass[i].reg) &
		     s5_tvled_dv_bypass[i].mask) != s5_tvled_dv_bypass[i].bypass)
			return false;
	return true;
}

/* display lock held. No NULL VSIF; the normal GUI modeset owns packet reset. */
static int tvled_display_end(struct dvfel_dev *d)
{
	int ret;

	tvled_display_quiesce(d);
	tvled_signal_release(d);
	ret = tvled_attr_update(false);
	if (ret)
		return ret;
	if (d->display_native_saved)
		set_amdv_enable(d->display_saved_enable);
	WRITE_ONCE(tvled_vsif, d->display_saved_vsif);
	d->display_native_saved = false;
	d->display_owner = NULL;
	return 0;
}

static int tvled_display_prepare(struct dvfel_dev *d)
{
	bool enable, first;
	int ret;

	tvled_display_quiesce(d);
	/* Do not disable native processing before its own BYPASS has completed. */
	if (!tvled_native_inactive())
		return -EBUSY;
	tvled_signal_release(d);
	WRITE_ONCE(tvled_vsif, false);
	enable = is_amdv_enable();
	first = !d->display_native_saved;
	if (first) {
		d->display_saved_enable = enable;
		d->display_native_saved = true;
	}
	set_amdv_enable(false);
	ret = tvled_attr_update(true);
	if (ret) {
		set_amdv_enable(enable);
		if (first && !tvled_attr_saved)
			d->display_native_saved = false;
		return ret;
	}
	if (!++d->display_epoch)
		++d->display_epoch;
	d->display_prepared = true;
	return 0;
}

static int tvled_display_ready(struct dvfel_dev *d)
{
	struct hdmitx_dev *hdev = get_hdmitx21_device();
	struct hdmitx_common *tx;
	unsigned long flags;
	int ret = 0;

	if (!d->display_prepared)
		return -EAGAIN;
	if (READ_ONCE(d->display_ready))
		return 0;
	if (!hdev)
		return -ENODEV;
	tx = &hdev->tx_comm;
	mutex_lock(&tvled_signal_lock);
	mutex_lock(&tx->hdmimode_mutex);
	if (!tx->ready || !tx->hpd_state || hdev->bist_lock ||
	    tx->fmt_para.cs != HDMI_COLORSPACE_RGB ||
	    tx->fmt_para.cd != COLORDEPTH_24B ||
	    tx->fmt_para.timing.h_active != d->display_width ||
	    tx->fmt_para.timing.v_active != d->display_height ||
	    is_amdv_enable() || !tvled_native_inactive()) {
		ret = -EAGAIN;
		goto out;
	}
	if (!hdev->tx_hw.base.debugfun) {
		ret = -EOPNOTSUPP;
		goto out;
	}
	/* Fixed S5 backend commands, not userspace input. The vendor wh branch
	 * writes/readbacks TX bytes without taking hdmimode_mutex itself.
	 * CONFIG_CSC/CSC_RGB_8BIT would instead enable YUV->RGB (0x0865).
	 */
	hdev->tx_hw.base.debugfun(&hdev->tx_hw.base, "whd20 0");
	hdev->tx_hw.base.debugfun(&hdev->tx_hw.base, "whd21 8");
	hdev->tx_hw.base.debugfun(&hdev->tx_hw.base, "whda0 0");
	hdev->tx_hw.base.debugfun(&hdev->tx_hw.base, "whda1 8");
	WRITE_ONCE(tvled_vsif, true);
	if (!tvled_signal_acquire_locked(d, 0, true)) {
		WRITE_ONCE(tvled_vsif, false);
		tvled_signal_state(d, false);
		ret = -EIO;
		goto out;
	}
	spin_lock_irqsave(&d->lock, flags);
	d->display_ready = true;
	spin_unlock_irqrestore(&d->lock, flags);
 out:
	mutex_unlock(&tx->hdmimode_mutex);
	mutex_unlock(&tvled_signal_lock);
	return ret;
}

static long ioc_display(struct dvfel_dev *d, struct file *f, void __user *arg)
{
	struct dvfel_display request;
	int ret = 0;

	if (copy_from_user(&request, arg, sizeof(request)))
		return -EFAULT;
	if (request.flags || !request.width || !request.height ||
	    request.width > MAX_W || request.height > MAX_H ||
	    ((request.width | request.height) & 1))
		return -EINVAL;
	mutex_lock(&tvled_display_lock);
	if (READ_ONCE(d->stopping)) {
		ret = -ENODEV;
		goto out;
	}
	if (request.op == DVFEL_DISPLAY_BEGIN) {
		if (request.session || request.epoch) {
			ret = -EINVAL;
			goto out;
		}
		if (!is_meson_s5_cpu() || !READ_ONCE(tvled)) {
			ret = -EOPNOTSUPP;
			goto out;
		}
		if (d->display_owner || tvled_attr_saved || READ_ONCE(d->signal_owned)) {
			ret = -EBUSY;
			goto out;
		}
		if (!++d->display_sequence)
			++d->display_sequence;
		d->display_session = d->display_sequence;
		d->display_epoch = 1;
		d->display_width = request.width;
		d->display_height = request.height;
		d->display_saved_vsif = READ_ONCE(tvled_vsif);
		d->display_owner = f;
		WRITE_ONCE(d->display_lifecycle, true);
		tvled_display_quiesce(d);
	} else {
		if (d->display_owner != f) {
			ret = -EPERM;
			goto out;
		}
		if (request.session != d->display_session || request.epoch != d->display_epoch) {
			ret = -ESTALE;
			goto out;
		}
		if (request.width != d->display_width || request.height != d->display_height) {
			ret = -EINVAL;
			goto out;
		}
		switch (request.op) {
		case DVFEL_DISPLAY_PREPARE:
			ret = tvled_display_prepare(d);
			break;
		case DVFEL_DISPLAY_READY:
			ret = tvled_display_ready(d);
			break;
		case DVFEL_DISPLAY_END:
			ret = tvled_display_end(d);
			break;
		default:
			ret = -EINVAL;
		}
	}
	if (!ret) {
		request.session = d->display_session;
		request.epoch = d->display_epoch;
		if (copy_to_user(arg, &request, sizeof(request))) {
			if (d->display_owner == f)
				tvled_display_end(d);
			ret = -EFAULT;
		}
	}
 out:
	mutex_unlock(&tvled_display_lock);
	return ret;
}


/* ------------------------------------------------------------------ */
/* dvfel owned buffers                                                */
/* ------------------------------------------------------------------ */

/* the video keeper holds the frame of this slot */
static bool slot_kept(struct dvfel_slot *s)
{
	return (s->head_mm && atomic_read(&s->head_mm->use_cnt) > 1) ||
	       (s->linear_mm && atomic_read(&s->linear_mm->use_cnt) >
			(s->pool_mem ? 2 : 1));
}

/*
 * The body goes with the header: when the keeper holds the header of a
 * frame on screen past our release, the body is released after it.
 */
static atomic_t bodies_pending = ATOMIC_INIT(0);

static void body_release_cb(struct codec_mm_s *head, struct codec_mm_cb_s *cb)
{
	codec_mm_release(cb->private_data, DRV_NAME);
	kfree(cb);
	atomic_dec(&bodies_pending);
}

static int tie_body_to_head(struct dvfel_slot *s)
{
	struct codec_mm_cb_s *cb = kzalloc(sizeof(*cb), GFP_KERNEL);

	if (!cb)
		return -ENOMEM;
	cb->func = body_release_cb;
	cb->private_data = s->body_mm;
	atomic_inc(&bodies_pending);
	codec_mm_add_release_callback(s->head_mm, cb);
	return 0;
}

/* a held slot may be reused (d->lock held) */
static bool slot_released_locked(struct dvfel_dev *d, struct dvfel_slot *s)
{
	/* Last exporter release proves there can be no remaining GPU access to
	 * an orphan's output allocation. It only updates allocation metadata;
	 * no asynchronous callback ever touches this driver/slot.
	 */
	if (s->pool_mem && s->pool_writer &&
	    s->pool_mem->orphan_writer == s->pool_writer &&
	    !atomic_read(&s->pool_mem->exporters)) {
		s->pool_writer = 0;
		s->pool_mem->orphan_writer = 0;
		s->state = SLOT_FREE;
	}
	if (s->pool_mem && (s->pool_writer || slot_kept(s)))
		return false;
	if (s->state == SLOT_HELD && d->shown_new && !slot_kept(s)) {
		s->state = SLOT_FREE;
		s->orig = NULL;
	}
	return s->state == SLOT_FREE;
}

/*
 * After a reset every slot may still be held by the display: it had taken
 * them all (e.g. waiting on a clock that jumped at a chapter skip). Held
 * slots are released once a frame of the new stream went out, which itself
 * needs a slot. When nothing of the new stream is on its way, stage A
 * passes the next frame through instead (d->lock held).
 */
static bool held_deadlock_locked(struct dvfel_dev *d)
{
	int i;

	if (d->shown_new || d->pend_cnt || d->outq_cnt)
		return false;
	for (i = 0; i < d->nslots; i++)
		if (d->slots[i].state == SLOT_FREE || d->slots[i].state == SLOT_BUSY)
			return false;
	return true;
}

/* some slot may still be on screen (d->lock held) */
static bool slots_shown_locked(struct dvfel_dev *d)
{
	int i;

	for (i = 0; i < d->nslots; i++) {
		struct dvfel_slot *s = &d->slots[i];

		if (s->state == SLOT_OUT || slot_kept(s) ||
		    (s->state == SLOT_HELD && !slot_released_locked(d, s)))
			return true;
	}
	return false;
}

/* buf_lock held. Detach under the vframe lock before dropping references:
 * a late display put may otherwise read use_cnt through a freed slot handle.
 */
static void dvfel_free_bufs(struct dvfel_dev *d)
{
	int i;
	unsigned long irqflags;

	spin_lock_irqsave(&d->lock, irqflags);
	d->pool_ready = false;
	pool_next_token_locked(d);
	spin_unlock_irqrestore(&d->lock, irqflags);

	/* The only submitter is stage B under buf_lock, using blocking blits. */
	if (d->ge2d) {
		destroy_ge2d_work_queue(d->ge2d);
		d->ge2d = NULL;
	}
	d->ge2d_verified = d->ge2d_failed = false;
	d->ge2d_copies = 0;
	for (i = 0; i < d->nslots; i++) {
		struct dvfel_slot *s = &d->slots[i];
		struct codec_mm_s *head, *body, *linear;
		struct dvfel_pool_mem *pool;
		ulong table;
		unsigned long flags;

		spin_lock_irqsave(&d->lock, flags);
		head = s->head_mm;
		/* A tied body remains owned by the header's release callback. */
		body = s->body_tied ? NULL : s->body_mm;
		linear = s->linear_mm;
		pool = s->pool_mem;
		table = s->table_handle;
		s->head_mm = s->body_mm = s->linear_mm = NULL;
		s->pool_mem = NULL;
		s->pool_writer = 0;
		s->body_tied = false;
		s->table_handle = s->table_phys = 0;
		s->head_phys = s->body_phys = s->linear_phys = 0;
		s->linear_ge2d_prepared = false;
		s->orig = NULL;
		s->taken = false;
		s->state = SLOT_FREE;
		spin_unlock_irqrestore(&d->lock, flags);

		/* These operations may sleep or invoke a codec_mm release callback. */
		if (table)
			codec_mm_dma_free_coherent(table);
		if (head)
			codec_mm_release(head, DRV_NAME);
		if (body)
			codec_mm_release(body, DRV_NAME);
		if (linear)
			codec_mm_release(linear, DRV_NAME);
		pool_mem_put(pool);
	}
	if (d->lin_phys)
		codec_mm_free_for_dma(DRV_NAME, d->lin_phys);
	d->lin_phys = 0;
	d->bufs_ok = false;
}

static ulong alloc_dma(u32 size)
{
	return codec_mm_alloc_for_dma(DRV_NAME, PAGE_ALIGN(size) / PAGE_SIZE, 0,
				      CODEC_MM_FLAGS_DMA);
}

static struct codec_mm_s *alloc_mm(u32 size, ulong *phys)
{
	struct codec_mm_s *mm = codec_mm_alloc(DRV_NAME, PAGE_ALIGN(size), 0,
					       CODEC_MM_FLAGS_DMA);

	*phys = mm ? mm->phy_addr : 0;
	return mm;
}

#define POOL_OWNER "dvfel_pool"

static struct dvfel_pool_mem *pool_mem_new(struct codec_mm_s *mm, ulong phys,
					 size_t size)
{
	struct dvfel_pool_mem *p = kzalloc(sizeof(*p), GFP_KERNEL);

	if (!p)
		return NULL;
	if ((phys & ~PAGE_MASK) || phys + PAGE_ALIGN(size) < phys ||
	    phys + PAGE_ALIGN(size) > 0x100000000ULL ||
	    codec_mm_request_shared_mem(mm, POOL_OWNER)) {
		kfree(p);
		return NULL;
	}
	refcount_set(&p->refs, 1);
	atomic_set(&p->exporters, 0);
	p->mm = mm;
	p->phys = phys;
	p->size = PAGE_ALIGN(size);
	return p;
}

static void pool_mem_put(struct dvfel_pool_mem *p)
{
	if (p && refcount_dec_and_test(&p->refs)) {
		codec_mm_release(p->mm, POOL_OWNER);
		kfree(p);
	}
}

static int pool_dma_attach(struct dma_buf *db, struct dma_buf_attachment *att)
{
	struct dvfel_pool_export *e = db->priv;
	struct dvfel_pool_attachment *a = kzalloc(sizeof(*a), GFP_KERNEL);
	int ret;

	if (!a)
		return -ENOMEM;
	ret = sg_alloc_table(&a->table, 1, GFP_KERNEL);
	if (ret) {
		kfree(a);
		return ret;
	}
	sg_set_page(a->table.sgl, pfn_to_page(PHYS_PFN(e->mem->phys)), e->mem->size, 0);
	a->dev = att->dev;
	att->priv = a;
	mutex_lock(&e->lock);
	list_add_tail(&a->node, &e->attachments);
	mutex_unlock(&e->lock);
	return 0;
}

static void pool_dma_detach(struct dma_buf *db, struct dma_buf_attachment *att)
{
	struct dvfel_pool_export *e = db->priv;
	struct dvfel_pool_attachment *a = att->priv;

	mutex_lock(&e->lock);
	list_del(&a->node);
	WARN_ON(a->mapped);
	mutex_unlock(&e->lock);
	sg_free_table(&a->table);
	kfree(a);
}

static struct sg_table *pool_dma_map(struct dma_buf_attachment *att,
				    enum dma_data_direction direction)
{
	struct dvfel_pool_attachment *a = att->priv;
	struct dvfel_pool_export *e = att->dmabuf->priv;
	int ret;

	mutex_lock(&e->lock);
	if (a->mapped) {
		ret = -EBUSY;
		goto out;
	}
	ret = dma_map_sgtable(att->dev, &a->table, direction, DMA_ATTR_SKIP_CPU_SYNC);
	if (!ret)
		a->mapped = true;
out:
	mutex_unlock(&e->lock);
	return ret ? ERR_PTR(ret) : &a->table;
}

static void pool_dma_unmap(struct dma_buf_attachment *att, struct sg_table *table,
			   enum dma_data_direction direction)
{
	struct dvfel_pool_attachment *a = att->priv;
	struct dvfel_pool_export *e = att->dmabuf->priv;

	mutex_lock(&e->lock);
	dma_unmap_sgtable(att->dev, table, direction, DMA_ATTR_SKIP_CPU_SYNC);
	a->mapped = false;
	mutex_unlock(&e->lock);
}

static int pool_dma_begin_cpu(struct dma_buf *db, enum dma_data_direction direction)
{
	struct dvfel_pool_export *e = db->priv;
	struct dvfel_pool_attachment *a;

	mutex_lock(&e->lock);
	list_for_each_entry(a, &e->attachments, node)
		if (a->mapped)
			dma_sync_sgtable_for_cpu(a->dev, &a->table, direction);
	mutex_unlock(&e->lock);
	return 0;
}

static int pool_dma_end_cpu(struct dma_buf *db, enum dma_data_direction direction)
{
	struct dvfel_pool_export *e = db->priv;
	struct dvfel_pool_attachment *a;

	mutex_lock(&e->lock);
	list_for_each_entry(a, &e->attachments, node)
		if (a->mapped)
			dma_sync_sgtable_for_device(a->dev, &a->table, direction);
	mutex_unlock(&e->lock);
	return 0;
}

static int pool_dma_mmap(struct dma_buf *db, struct vm_area_struct *vma)
{
	struct dvfel_pool_export *e = db->priv;
	unsigned long pages = e->mem->size >> PAGE_SHIFT;
	unsigned long length = vma->vm_end - vma->vm_start;

	if (vma->vm_pgoff >= pages || length > (pages - vma->vm_pgoff) << PAGE_SHIFT)
		return -EINVAL;
	vma->vm_flags |= VM_DONTEXPAND | VM_DONTDUMP;
	vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
	return remap_pfn_range(vma, vma->vm_start,
			       PHYS_PFN(e->mem->phys) + vma->vm_pgoff,
			       length, vma->vm_page_prot);
}

static void pool_dma_release(struct dma_buf *db)
{
	struct dvfel_pool_export *e = db->priv;

	WARN_ON(!list_empty(&e->attachments));
	atomic_dec(&e->mem->exporters);
	pool_mem_put(e->mem);
	kfree(e);
	/* dma-buf core drops its .owner module pin only after we return. */
}

static const struct dma_buf_ops pool_dma_ops = {
	.attach = pool_dma_attach,
	.detach = pool_dma_detach,
	.map_dma_buf = pool_dma_map,
	.unmap_dma_buf = pool_dma_unmap,
	.begin_cpu_access = pool_dma_begin_cpu,
	.end_cpu_access = pool_dma_end_cpu,
	.mmap = pool_dma_mmap,
	.release = pool_dma_release,
};

/* buf_lock held. The slot's metadata reference keeps p alive while exported.
 * Slots never cache the resulting dma-buf or its module reference.
 */
static struct dma_buf *pool_export(struct dvfel_pool_mem *p)
{
	DEFINE_DMA_BUF_EXPORT_INFO(info);
	struct dvfel_pool_export *e = kzalloc(sizeof(*e), GFP_KERNEL);
	struct dma_buf *db;

	if (!e)
		return ERR_PTR(-ENOMEM);
	mutex_init(&e->lock);
	INIT_LIST_HEAD(&e->attachments);
	e->mem = p;
	refcount_inc(&p->refs);
	atomic_inc(&p->exporters);
	info.ops = &pool_dma_ops;
	info.size = p->size;
	info.flags = O_RDWR;
	info.priv = e;
	db = dma_buf_export(&info);
	if (IS_ERR(db)) {
		atomic_dec(&p->exporters);
		pool_mem_put(p);
		kfree(e);
	}
	return db;
}

static int dvfel_alloc_bufs(struct dvfel_dev *d, u32 w, u32 h)
{
	int i;

	d->lin_phys = alloc_dma(LIN_SIZE(w, h));
	if (!d->lin_phys)
		goto fail;

	for (i = 0; i < d->nslots; i++) {
		struct dvfel_slot *s = &d->slots[i];
		u32 body = FBC_BODY_SIZE(w, h), j, *tbl;
		ulong phys;

		s->body_mm = alloc_mm(body, &s->body_phys);
		s->head_mm = alloc_mm(FBC_HEAD_SIZE(w, h), &s->head_phys);
		tbl = codec_mm_dma_alloc_coherent(&s->table_handle, &phys,
						  FBC_TABLE_SIZE(w, h), DRV_NAME);
		if (!s->body_phys || !s->head_phys || !tbl)
			goto fail;
		if (tie_body_to_head(s))
			goto fail;
		s->body_tied = true;
		/* the AFBCE MMU table holds 20-bit page numbers */
		if (s->body_phys + body > 0x100000000ULL)
			goto fail;
		s->table_phys = phys;
		for (j = 0; j < body; j += PAGE_SIZE)
			*tbl++ = ((s->body_phys + j) >> PAGE_SHIFT) & 0xfffff;
	}
	d->buf_w = w;
	d->buf_h = h;
	d->bufs_ok = true;
	pr_info(DRV_NAME ": buffers allocated for %ux%u (lin %u KiB, fbc %u KiB x%d)\n",
		w, h, LIN_SIZE(w, h) >> 10,
		(FBC_BODY_SIZE(w, h) + FBC_HEAD_SIZE(w, h)) >> 10, d->nslots);
	return 0;
fail:
	pr_err(DRV_NAME ": buffer allocation failed\n");
	dvfel_free_bufs(d);
	return -ENOMEM;
}

static bool dvfel_idle_locked(struct dvfel_dev *d)
{
	int i;

	if (d->prov_reg)
		return false;
	for (i = 0; i < d->nslots; i++)
		if (d->slots[i].state == SLOT_BUSY)
			return false;
	return true;
}

static void dvfel_free_work(struct work_struct *work)
{
	struct dvfel_dev *d = container_of(to_delayed_work(work),
					   struct dvfel_dev, free_work);
	unsigned long flags;
	bool idle;

	/* REG cancels this work before setting prov_reg, but recheck after
	 * taking buf_lock so a stale idle snapshot never authorizes freeing.
	 */
	mutex_lock(&d->buf_lock);
	spin_lock_irqsave(&d->lock, flags);
	idle = dvfel_idle_locked(d);
	spin_unlock_irqrestore(&d->lock, flags);

	/* a frame the keeper still shows outlives this, see tie_body_to_head() */
	if (idle && d->bufs_ok) {
		dvfel_free_bufs(d);
		pr_info(DRV_NAME ": buffers released\n");
	}
	mutex_unlock(&d->buf_lock);
}

/* ------------------------------------------------------------------ */
/* queues and bookkeeping (d->lock held)                              */
/* ------------------------------------------------------------------ */

static void outq_push(struct dvfel_dev *d, struct vframe_s *vf)
{
	d->outq[(d->outq_head + d->outq_cnt) % QUEUE_LEN] = vf;
	d->outq_cnt++;
}

static struct vframe_s *outq_pop(struct dvfel_dev *d)
{
	struct vframe_s *vf;

	if (!d->outq_cnt)
		return NULL;
	vf = d->outq[d->outq_head];
	d->outq_head = (d->outq_head + 1) % QUEUE_LEN;
	d->outq_cnt--;
	return vf;
}

static void pend_push(struct dvfel_dev *d, struct vframe_s *vf, int job, u32 gen)
{
	struct dvfel_pend *p = &d->pendq[(d->pend_head + d->pend_cnt) % QUEUE_LEN];

	p->vf = vf;
	p->job = job;
	p->gen = gen;
	d->pend_cnt++;
}

static void pend_pop(struct dvfel_dev *d)
{
	d->pend_head = (d->pend_head + 1) % QUEUE_LEN;
	d->pend_cnt--;
}

static struct dvfel_slot *slot_of(struct dvfel_dev *d, struct vframe_s *vf)
{
	int i;

	for (i = 0; i < d->nslots; i++)
		if (&d->slots[i].out_vf == vf)
			return &d->slots[i];
	return NULL;
}

static struct dvfel_slot *slot_free(struct dvfel_dev *d)
{
	int i;

	for (i = 0; i < d->nslots; i++)
		if (slot_released_locked(d, &d->slots[i]))
			return &d->slots[i];
	return NULL;
}

static struct dvfel_pass *pass_free(struct dvfel_dev *d)
{
	int i;

	for (i = 0; i < QUEUE_LEN; i++)
		if (!d->pass[i].vf)
			return &d->pass[i];
	return NULL;
}

static int job_free(struct dvfel_dev *d)
{
	int i;

	for (i = 0; i < d->client_nbufs; i++)
		if (d->jobs[i].state == JOB_FREE)
			return i;
	return -1;
}

/* d->lock held. DONE/UNREG is the producer's GPU-completion promise. A taken
 * direct pool job that times out retains its slot while the GPU may write it.
 */
static void pool_writer_done_locked(struct dvfel_kjob *j)
{
	struct dvfel_slot *s = j->slot;

	if ((j->flags & DVFEL_JOB_TVLED_POOL) && s && s->pool_writer == j->id) {
		s->pool_writer = 0;
		if (s->pool_mem && s->pool_mem->orphan_writer == j->id)
			s->pool_mem->orphan_writer = 0;
		if (j->orphan)
			s->state = SLOT_FREE;
	}
}

/* release a job whose buffers stage B no longer needs */
static void job_release(struct dvfel_dev *d, struct dvfel_kjob *j)
{
	if (j->state == JOB_TAKEN) {
		/* the compositor still writes out[]: free on JOB_DONE/close */
		j->orphan = true;
		if ((j->flags & DVFEL_JOB_TVLED_POOL) && j->slot &&
		    j->slot->pool_writer == j->id) {
			j->slot->state = SLOT_BUSY;
			if (j->slot->pool_mem)
				j->slot->pool_mem->orphan_writer = j->id;
			j->slot->orig = NULL;
			j->orig = NULL;
			return; /* retain output slot as well as the input pair */
		}
	} else {
		pool_writer_done_locked(j);
		j->state = JOB_FREE;
		j->orphan = false;
	}
	j->slot = NULL;
	j->orig = NULL;
}

/* drop everything we hold; frames are not returned upstream */
static void dvfel_reset_locked(struct dvfel_dev *d)
{
	int i;

	d->gen++;
	d->new_stream = true;
	d->shown_new = false;
	d->tvled_started = false;
	d->tvled_first_delivery = 0;
	d->tvled_have_pts = false;
	d->client_pool_token = 0;
	pool_next_token_locked(d);
	d->outq_head = d->outq_cnt = 0;
	/* pending entries of the old generation are dropped by stage B */
	for (i = 0; i < d->nslots; i++) {
		struct dvfel_slot *s = &d->slots[i];

		if (s->state == SLOT_OUT)
			s->state = s->taken ? SLOT_HELD : SLOT_FREE;
	}
	memset(d->pass, 0, sizeof(d->pass));
}

/* The S5 decoder calls these notifications in process context, after its
 * VFM provider table entry is closed and before freeing/reinitializing VFs.
 * Do not wait for queued GPU jobs: changing gen makes B retire those without
 * dereferencing orig. Only current pointer users must finish their work.
 */
static void bl_quiesce(struct dvfel_dev *d, bool reset_pending)
{
	unsigned long flags;

	spin_lock_irqsave(&d->lock, flags);
	d->prov_reg = false;
	d->bl_reset_pending = false;
	dvfel_reset_locked(d);
	spin_unlock_irqrestore(&d->lock, flags);
	d->kick_a = d->kick_b = true;
	wake_up_interruptible(&d->wq_a);
	wake_up_interruptible(&d->wq_b);
	wait_event(d->wq_bl_idle, !READ_ONCE(d->bl_users));
	spin_lock_irqsave(&d->lock, flags);
	d->bl_reset_pending = reset_pending;
	spin_unlock_irqrestore(&d->lock, flags);
}

/* ------------------------------------------------------------------ */
/* VICP                                                               */
/* ------------------------------------------------------------------ */

static bool frame_supported(struct vframe_s *vf)
{
	return (vf->type & VIDTYPE_COMPRESS) &&
	       (vf->bitdepth & BITDEPTH_YMASK) == BITDEPTH_Y10 &&
	       !(vf->type & VIDTYPE_TYPEMASK) &&	/* progressive only */
	       vf->compWidth && vf->compHeight &&
	       vf->compWidth <= MAX_W && vf->compHeight <= MAX_H &&
	       !(vf->compWidth & 1) && !(vf->compHeight & 1);
}

/*
 * The write MIF derives its line stride from the background width assuming
 * 30 bits per pixel, but writes 32 bits per pixel in 10-bit 4:4:4 mode.
 * Pick the background width whose computed stride is exactly 4 * w.
 */
static u32 wmif_bg_width(u32 w)
{
	u32 bw;

	for (bw = w; bw <= w * 2; bw++)
		if (ALIGN(DIV_ROUND_UP(bw * 30, 128) * 16, 64) == LIN_STRIDE(w))
			return bw;
	return 0;
}

/* (A) AFBC frame -> linear 10-bit 4:4:4 at @dst */
static int vicp_decompress(struct vicp_ctx *c, struct vframe_s *vf,
			   ulong dst, u32 w, u32 h)
{
	u32 bw = wmif_bg_width(w);

	if (!bw)
		return -EINVAL;
	/*
	 * VICP enables its HDR->SDR block for any non-SDR source format and
	 * clears fgs_valid on its input: hand it a private copy with the
	 * source format neutralized.
	 */
	c->vin = *vf;
	/*
	 * A picture coded taller or wider than shown (1920x1088 cropped to
	 * 1080 by the SPS conformance window, common for an EL): read only
	 * the top left w x h of it. The AFBC headers are laid out block row
	 * by block row, so fewer rows is just a shorter read; the output
	 * axis equal to the source size keeps the scaler out.
	 */
	if (w < c->vin.compWidth)
		c->vin.compWidth = w;
	if (h < c->vin.compHeight)
		c->vin.compHeight = h;
	c->vin.src_fmt.sei_magic_code = 0;
	c->vin.src_fmt.fmt = VFRAME_SIGNAL_FMT_INVALID;
	c->vin.fgs_valid = false;

	memset(&c->cfg, 0, sizeof(c->cfg));
	c->cfg.input_data.is_vframe = true;
	c->cfg.input_data.data_vf = &c->vin;
	c->cfg.output_data.phy_addr[0] = dst;
	c->cfg.output_data.width = bw;
	c->cfg.output_data.height = h;
	c->cfg.output_data.endian = 1;
	c->cfg.output_data.mif_out_en = 1;
	c->cfg.output_data.mif_color_fmt = VICP_COLOR_FORMAT_YUV444;
	c->cfg.output_data.mif_color_dep = 10;
	c->cfg.output_data.fbc_out_en = 0;
	c->cfg.output_data.fbc_color_fmt = VICP_COLOR_FORMAT_YUV420;
	c->cfg.output_data.fbc_color_dep = 10;
	c->cfg.output_data.out_sig_fmt = VFRAME_SIGNAL_FMT_SDR;
	c->cfg.data_option.rotation_mode = VICP_ROTATION_0;
	c->cfg.data_option.output_axis.width = w;
	c->cfg.data_option.output_axis.height = h;
	return vicp_process(&c->cfg);
}

/* (B) linear 10-bit 4:4:4 at @src -> AFBC 10-bit 4:2:0 owned by the slot */
static int vicp_compress(struct vicp_ctx *c, ulong src, struct dvfel_slot *s,
			 u32 w, u32 h)
{
	memset(&c->dma, 0, sizeof(c->dma));
	c->dma.buf_addr = src;
	c->dma.buf_stride_w = LIN_STRIDE(w);	/* bytes */
	c->dma.buf_stride_h = h;
	c->dma.data_width = w;
	c->dma.data_height = h;
	c->dma.plane_count = 1;
	c->dma.color_format = VICP_COLOR_FORMAT_YUV444;
	c->dma.color_depth = 10;
	c->dma.endian = 1;

	memset(&c->cfg, 0, sizeof(c->cfg));
	c->cfg.input_data.is_vframe = false;
	c->cfg.input_data.data_dma = &c->dma;
	c->cfg.output_data.phy_addr[0] = s->body_phys;
	c->cfg.output_data.phy_addr[1] = s->head_phys;
	c->cfg.output_data.phy_addr[2] = s->table_phys;
	c->cfg.output_data.width = w;
	c->cfg.output_data.height = h;
	c->cfg.output_data.endian = 1;
	c->cfg.output_data.mif_out_en = 0;
	c->cfg.output_data.mif_color_fmt = VICP_COLOR_FORMAT_YUV444;
	c->cfg.output_data.mif_color_dep = 10;
	c->cfg.output_data.fbc_out_en = 1;
	c->cfg.output_data.fbc_color_fmt = VICP_COLOR_FORMAT_YUV420;
	c->cfg.output_data.fbc_color_dep = 10;
	c->cfg.output_data.fbc_init_ctrl = 1;
	c->cfg.output_data.fbc_pip_mode = 1;
	c->cfg.output_data.out_sig_fmt = VFRAME_SIGNAL_FMT_SDR;
	c->cfg.data_option.rotation_mode = VICP_ROTATION_0;
	c->cfg.data_option.output_axis.width = w;
	c->cfg.data_option.output_axis.height = h;
	return vicp_process(&c->cfg);
}

/* vicp_process() reports 0 even on a 200 ms hardware timeout */
static bool vicp_ok(struct dvfel_dev *d, int ret, s64 us, const char *what)
{
	if (!ret && us < 190000)
		return true;
	d->vicp_err++;
	pr_err(DRV_NAME ": VICP %s failed (ret %d, %lld us)\n", what, ret, us);
	return false;
}

static void build_out_vf(struct dvfel_slot *s, struct vframe_s *orig, u32 w, u32 h)
{
	struct vframe_s *o = &s->out_vf;

	/* keep index, timing, signal and source info of the decoder frame */
	*o = *orig;
	INIT_LIST_HEAD(&o->list);
	o->type = VIDTYPE_PROGRESSIVE | VIDTYPE_VIU_FIELD | VIDTYPE_COMPRESS |
		  VIDTYPE_SCATTER | VIDTYPE_NO_DW;
	o->type_backup = o->type;
	o->type_original = o->type;
	o->bitdepth = BITDEPTH_Y10 | BITDEPTH_U10 | BITDEPTH_V10;
	o->flag |= VFRAME_FLAG_COMPOSER_DONE;
	o->compHeadAddr = s->head_phys;
	o->compBodyAddr = s->body_phys;
	o->compWidth = w;
	o->compHeight = h;
	o->width = w;
	o->height = h;
	/* AFBC only, like a decoder without double write: no MIF canvases */
	o->canvas0Addr = 0;
	o->canvas1Addr = 0;
	o->plane_num = 0;
	memset(o->canvas0_config, 0, sizeof(o->canvas0_config));
	memset(o->canvas1_config, 0, sizeof(o->canvas1_config));
	/*
	 * the video keeper holds the header by its handle (so the display
	 * keeps showing this frame across a decoder reset); the body is a
	 * scatter-type AFBC body to it, so it goes with the header instead
	 */
	o->mem_handle = NULL;
	o->mem_handle_1 = NULL;
	o->mem_head_handle = s->head_mm;
	o->mem_dw_handle = NULL;
	o->vf_ext = NULL;
	o->uvm_vf = NULL;
	o->early_process_fun = NULL;
	o->process_fun = NULL;
	o->private_data = NULL;
	o->fence = NULL;
	o->fgs_valid = false;
}

/* The display's 422 "old mode" consumes two 12-bit words per pixel.
 * Keep one allocation per output slot and expose its codec_mm handle to the
 * video keeper, so a decoder reset cannot free a picture still on screen.
 */
static bool linear12_slot(struct dvfel_dev *d, struct dvfel_slot *s, u32 w, u32 h)
{
	if (s->pool_mem && READ_ONCE(s->pool_mem->abandoned))
		return false;
	if (!s->linear_mm) {
		struct codec_mm_s *mm;
		struct dvfel_pool_mem *pool = NULL;
		unsigned long flags;
		ulong phys;

		mm = alloc_mm(LINEAR12_SIZE(w, h), &phys);
		if (!mm)
			return false;
		if (tvled_pool) {
			pool = pool_mem_new(mm, phys, LINEAR12_SIZE(w, h));
			if (!pool) {
				codec_mm_release(mm, DRV_NAME);
				return false;
			}
		}
		/* Complete the fixed shared-ref metadata before exposing either
		 * pointer to slot_kept in interrupt context.
		 */
		spin_lock_irqsave(&d->lock, flags);
		s->linear_ge2d_prepared = false;
		s->linear_phys = phys;
		s->pool_mem = pool;
		s->linear_mm = mm;
		spin_unlock_irqrestore(&d->lock, flags);
	}
	return s->linear_mm != NULL;
}

/* buf_lock held. Exported output slots are independent of cin/cel/cout pairs. */
static bool pool_prepare(struct dvfel_dev *d, u32 w, u32 h)
{
	unsigned long flags;
	int i;

	if (d->nslots > DVFEL_POOL_MAX_SLOTS)
		return false;
	for (i = 0; i < d->nslots; i++)
		if (!linear12_slot(d, &d->slots[i], w, h) || !d->slots[i].pool_mem)
			return false;
	spin_lock_irqsave(&d->lock, flags);
	if (!d->pool_ready) {
		pool_next_token_locked(d);
		d->pool_ready = true;
	}
	spin_unlock_irqrestore(&d->lock, flags);
	return true;
}

/* d->lock held; slot BUSY is reserved until publication or producer DONE. */
static bool pool_job_valid_locked(struct dvfel_dev *d, struct dvfel_kjob *j)
{
	struct dvfel_slot *s = j->slot;
	unsigned int index = DVFEL_JOB_POOL_SLOT(j->flags);

	return (j->flags & (DVFEL_JOB_TVLED | DVFEL_JOB_TVLED_POOL)) ==
		(DVFEL_JOB_TVLED | DVFEL_JOB_TVLED_POOL) && d->pool_ready &&
		j->gen == d->gen && j->pool_token == d->pool_token &&
		DVFEL_JOB_POOL_COOKIE(j->flags) == (d->pool_token & 0xffff) &&
		index < (unsigned int)d->nslots && index < DVFEL_POOL_MAX_SLOTS &&
		s == &d->slots[index] && s->state == SLOT_BUSY && s->pool_mem &&
		!s->pool_mem->abandoned &&
		s->linear_mm == s->pool_mem->mm && s->linear_phys == s->pool_mem->phys &&
		s->pool_mem->size >= LINEAR12_SIZE(d->buf_w, d->buf_h);
}

static void build_linear12_vf(struct dvfel_slot *s, struct vframe_s *orig, u32 w, u32 h)
{
	struct vframe_s *o = &s->out_vf;

	/* Preserve decoder index and Dolby source/RPU identity for native amdv. */
	*o = *orig;
	INIT_LIST_HEAD(&o->list);
	o->type = VIDTYPE_PROGRESSIVE | VIDTYPE_VIU_FIELD |
		  VIDTYPE_VIU_SINGLE_PLANE | VIDTYPE_VIU_422;
	o->type_backup = o->type;
	o->type_original = o->type;
	/* No FULL_PACK_422_MODE: bit_mode=1 is the 12-bit "old mode". */
	o->bitdepth = BITDEPTH_Y12 | BITDEPTH_U12 | BITDEPTH_V12;
	o->flag &= ~VFRAME_FLAG_DI_DW;
	o->flag |= VFRAME_FLAG_VIDEO_LINEAR | VFRAME_FLAG_COMPOSER_DONE;
	o->compHeadAddr = 0;
	o->compBodyAddr = 0;
	o->compWidth = w;
	o->compHeight = h;
	o->width = w;
	o->height = h;
	o->canvas0Addr = (u32)-1;
	o->canvas1Addr = (u32)-1;
	o->plane_num = 1;
	memset(o->canvas0_config, 0, sizeof(o->canvas0_config));
	memset(o->canvas1_config, 0, sizeof(o->canvas1_config));
	o->canvas0_config[0].phy_addr = s->linear_phys;
	o->canvas0_config[0].width = LINEAR12_STRIDE(w);
	o->canvas0_config[0].height = h;
	memcpy(o->canvas1_config, o->canvas0_config, sizeof(o->canvas1_config));
	o->mem_handle = s->linear_mm;
	o->mem_handle_1 = NULL;
	o->mem_head_handle = NULL;
	o->mem_dw_handle = NULL;
	o->vf_ext = NULL;
	o->uvm_vf = NULL;
	o->early_process_fun = NULL;
	o->process_fun = NULL;
	o->private_data = NULL;
	o->fence = NULL;
	o->fgs_valid = false;
}

/* RGB888 here is an already packed transport, never an RPU-bearing
 * decoder picture. Remove the Dolby identity and HDR signalling so the
 * native core cannot reshape it; all carrier bytes are full-range values.
 */
static void build_tvled_vf(struct dvfel_slot *s, struct vframe_s *orig, u32 w, u32 h)
{
	struct vframe_s *o = &s->out_vf;

	build_linear12_vf(s, orig, w, h);
	o->type = VIDTYPE_PROGRESSIVE | VIDTYPE_VIU_FIELD | VIDTYPE_VIU_SINGLE_PLANE |
		  VIDTYPE_VIU_444 | VIDTYPE_RGB_444;
	o->type_backup = o->type;
	o->type_original = o->type;
	o->bitdepth = BITDEPTH_Y8 | BITDEPTH_U8 | BITDEPTH_V8;
	/* video available, unspecified format, full range, BT.709 SDR, RGB matrix */
	o->signal_type = (1 << 29) | (5 << 26) | (1 << 25) | (1 << 24) |
			 (1 << 16) | (1 << 8);
	o->ext_signal_type = 0;
	memset(&o->prop.master_display_colour, 0, sizeof(o->prop.master_display_colour));
	o->hdr10p_data_size = 0;
	o->hdr10p_data_buf = NULL;
	o->meta_data_size = 0;
	o->meta_data_buf = NULL;
	memset(&o->src_fmt, 0, sizeof(o->src_fmt));
	o->src_fmt.sei_magic_code = SEI_MAGIC_CODE;
	o->src_fmt.fmt = VFRAME_SIGNAL_FMT_SDR;
	o->discard_dv_data = true;
	o->dv_input = false;
	o->enhance_vf = NULL;
}

/* buf_lock held. GPU completion was explicitly reported by userspace; the
 * display keeps using the established real linear_mm keeper handle.
 */
static bool publish_pool_frame(struct dvfel_dev *d, struct dvfel_kjob *j,
			       struct vframe_s *orig, u32 w, u32 h)
{
	unsigned long flags;
	bool ok;

	spin_lock_irqsave(&d->lock, flags);
	ok = pool_job_valid_locked(d, j) && !j->slot->pool_writer &&
		w == d->buf_w && h == d->buf_h;
	if (ok)
		build_tvled_vf(j->slot, orig, w, h);
	spin_unlock_irqrestore(&d->lock, flags);
	return ok;
}

static inline u32 pack24_word(u32 rgba, u32 s0, u32 s1, u32 s2)
{
	return ((rgba >> s0) & 0xff) |
	       (((rgba >> s1) & 0xff) << 8) |
	       (((rgba >> s2) & 0xff) << 16);
}

/* Input starts on a 4-byte boundary (RGBA stride w * 4), and output rows
 * are aligned to 64 bytes. Four pixels advance output by 12 bytes, keeping
 * every batch aligned. Word accesses avoid repeated uncached byte reads.
 */
static void compact_rgba24_row(u8 *out, const u8 *in, u32 w,
			       u32 s0, u32 s1, u32 s2)
{
	u32 x;

	for (x = 0; w - x >= 4; x += 4) {
		const __le32 *src = (const __le32 *)in;
		__le32 *dst = (__le32 *)out;
		u32 p0 = pack24_word(le32_to_cpu(READ_ONCE(src[0])), s0, s1, s2);
		u32 p1 = pack24_word(le32_to_cpu(READ_ONCE(src[1])), s0, s1, s2);
		u32 p2 = pack24_word(le32_to_cpu(READ_ONCE(src[2])), s0, s1, s2);
		u32 p3 = pack24_word(le32_to_cpu(READ_ONCE(src[3])), s0, s1, s2);

		WRITE_ONCE(dst[0], cpu_to_le32(p0 | (p1 << 24)));
		WRITE_ONCE(dst[1], cpu_to_le32((p1 >> 8) | (p2 << 16)));
		WRITE_ONCE(dst[2], cpu_to_le32((p2 >> 16) | (p3 << 8)));
		in += 16;
		out += 12;
	}
	for (; x < w; x++) {
		out[0] = in[s0 / 8];
		out[1] = in[s1 / 8];
		out[2] = in[s2 / 8];
		in += 4;
		out += 3;
	}
}

/* Validate the first copy and every 120th copy, including the source bytes.
 * All other GE2D frames leave the source exclusively in the device domain.
 */
static bool ge2d_carrier_golden(struct dvfel_cbuf *src, const u8 *dst, u32 w, u32 h)
{
	const u8 *cpu;
	bool equal = false;
	u32 x, y;

	if (dma_buf_begin_cpu_access(src->db, DMA_FROM_DEVICE))
		return false;
	cpu = src->map.vaddr;
	for (y = 0; y < h; y++) {
		const u8 *in = cpu + (size_t)y * LIN_STRIDE(w);
		const u8 *out = dst + (size_t)y * LINEAR12_STRIDE(w);

		for (x = 0; x < w; x++, in += 4, out += 3)
			if (out[0] != in[2] || out[1] != in[1] || out[2] != in[0])
				goto done;
		if ((y & 63) == 63)
			cond_resched();
	}
	equal = true;
 done:
	if (dma_buf_end_cpu_access(src->db, DMA_FROM_DEVICE))
		equal = false;
	return equal;
}

/* Synchronize complete CPU cache lines, including a tuple straddling a line.
 * The caller supplies raster-ordered offsets and a cache-aligned allocation.
 * codec_mm's virtual-to-physical lookup preserves each subrange's offset.
 */
static void ge2d_carrier_probe_sync(u8 *dst, const size_t *offsets, unsigned int count,
				   enum dma_data_direction direction)
{
	size_t align = dma_get_cache_alignment();
	size_t start = 0, end = 0;
	unsigned int i;

	for (i = 0; i < count; i++) {
		size_t next = round_down(offsets[i], align);
		size_t stop = ALIGN(offsets[i] + 3, align);

		if (i && next <= end) {
			end = max(end, stop);
			continue;
		}
		if (i)
			codec_mm_dma_flush(dst + start, end - start, direction);
		start = next;
		end = stop;
	}
	if (count)
		codec_mm_dma_flush(dst + start, end - start, direction);
}

/* buf_lock held; only order 0 calls this helper. The source remains owned
 * by this job until the blocking copy and its completion checks finish.
 */
static bool publish_ge2d_carrier(struct dvfel_dev *d, struct dvfel_slot *s,
				 struct dvfel_cbuf *src, u32 w, u32 h)
{
	struct config_para_ex_s cfg = { 0 };
	size_t offsets[6];
	u8 poison[6][3];
	u8 *dst;
	u32 stride = LINEAR12_STRIDE(w), x, y;
	u32 size = LINEAR12_SIZE(w, h), align = dma_get_cache_alignment();
	ktime_t started;
	long waited;
	bool prepared, partial, golden;
	int i;

	/* Direct GE2D strides are rounded to 8 bytes. Its CANVAS_ALLOC API
	 * expresses RGB24 row width in pixels, so reject non-integral /3.
	 */
	if (d->ge2d_failed || !w || !h || (LIN_STRIDE(w) & 7) || stride % 3 ||
	    ((src->phys | s->linear_phys) & 7))
		return false;
	if (!d->ge2d) {
		d->ge2d = create_ge2d_work_queue();
		if (!d->ge2d)
			return false;
	}
	dst = codec_mm_phys_to_virt(s->linear_phys);
	if (!dst)
		return false;
	/* Userspace has already glFinish()ed before JOB_DONE. Preserve the
	 * writer-fence wait from begin_cpu_access without invalidating the
	 * entire cached source just to inspect six pixels. A failed wait uses
	 * the existing CPU fallback, which retains its CPU-access bracketing.
	 */
	waited = dma_resv_wait_timeout(src->db->resv, false, true,
				       msecs_to_jiffies(100));
	if (waited <= 0) {
		d->ge2d_failed = true;
		pr_warn(DRV_NAME ": GE2D source writer wait failed (%ld); using CPU until buffers reset\n",
			waited);
		return false;
	}
	prepared = s->linear_ge2d_prepared;
	partial = IS_ALIGNED((unsigned long)dst, align) && IS_ALIGNED(size, align);
	golden = !d->ge2d_verified || !((d->ge2d_copies + 1) % 120);
	/* No outside CPU access is permitted once a prepared slot is reused.
	 * Any failure will enter the full CPU fallback and require preparation.
	 */
	s->linear_ge2d_prepared = false;
	/* The public blocking blit returns void, including on submit failure.
	 * Poison first/last pixels of the first/middle/last row to detect a
	 * missing/incomplete command without CPU-reading the source. A real
	 * pixel equal to a poison tuple conservatively triggers CPU fallback.
	 * This is a completion probe; periodic golden checks verify the bytes.
	 */
	for (i = 0; i < ARRAY_SIZE(offsets); i++) {
		u8 *out;

		y = i < 2 ? 0 : (i < 4 ? h / 2 : h - 1);
		x = (i & 1) ? w - 1 : 0;
		offsets[i] = (size_t)y * stride + x * 3;
		poison[i][0] = 0xa5 ^ i;
		poison[i][1] = 0x3c ^ (i << 2);
		poison[i][2] = 0x96 ^ (i << 4);
		out = dst + offsets[i];
		memcpy(out, poison[i], sizeof(poison[i]));
	}
	/* Padding is initialized only while the slot needs full preparation. */
	if (!prepared)
		for (y = 0; stride != w * 3 && y < h; y++)
			memset(dst + (size_t)y * stride + w * 3, 0, stride - w * 3);
	if (!prepared || golden || !partial)
		codec_mm_dma_flush(dst, size, DMA_BIDIRECTIONAL);
	else
		ge2d_carrier_probe_sync(dst, offsets, ARRAY_SIZE(offsets), DMA_BIDIRECTIONAL);

	/* Little endian ABGR32 = word AABBGGRR = GPU bytes R,G,B,A.
	 * Little endian RGB24 = word RRGGBB = display bytes B,G,R.
	 * The driver's V4L2 map likewise maps BGR24 to S24_RGB. Both formats
	 * use 8-bit channels; RGB-to-RGB disables every conversion matrix.
	 */
	cfg.src_para.mem_type = CANVAS_ALLOC;
	cfg.src_para.format = GE2D_FORMAT_S32_ABGR;
	cfg.src_para.width = w;
	cfg.src_para.height = h;
	cfg.src_planes[0].addr = src->phys;
	cfg.src_planes[0].w = w;
	cfg.src_planes[0].h = h;
	cfg.src2_para.mem_type = CANVAS_TYPE_INVALID;
	cfg.dst_para.mem_type = CANVAS_ALLOC;
	cfg.dst_para.format = GE2D_FORMAT_S24_RGB;
	cfg.dst_para.width = w;
	cfg.dst_para.height = h;
	cfg.dst_planes[0].addr = s->linear_phys;
	cfg.dst_planes[0].w = stride / 3;
	cfg.dst_planes[0].h = h;
	if (ge2d_context_config_ex(d->ge2d, &cfg) < 0) {
		d->ge2d_failed = true;
		pr_warn(DRV_NAME ": GE2D carrier configuration failed; using CPU until buffers reset\n");
		return false;
	}
	started = ktime_get();
	/* IS_BLIT/LOGIC_COPY: no alpha multiply, scaling, filtering or blend. */
	bitblt_noalpha(d->ge2d, 0, 0, w, h, 0, 0);
	/* A golden check reads every pixel. Other copies read only the probes. */
	if (golden || !partial)
		codec_mm_dma_flush(dst, size, DMA_FROM_DEVICE);
	else
		ge2d_carrier_probe_sync(dst, offsets, ARRAY_SIZE(offsets), DMA_FROM_DEVICE);
	/* The GE2D worker resets after its 1-second timeout but does not
	 * propagate status through bitblt_noalpha. Do not trust that frame.
	 */
	if (ktime_us_delta(ktime_get(), started) >= 900000) {
		d->ge2d_failed = true;
		pr_warn(DRV_NAME ": GE2D carrier copy took too long; using CPU until buffers reset\n");
		return false;
	}
	for (i = 0; i < ARRAY_SIZE(offsets); i++) {
		if (!memcmp(dst + offsets[i], poison[i], 3)) {
			d->ge2d_failed = true;
			pr_warn(DRV_NAME ": GE2D carrier completion probe unchanged; using CPU until buffers reset\n");
			return false;
		}
	}
	d->ge2d_copies++;
	if (golden) {
		bool equal = ge2d_carrier_golden(src, dst, w, h);

		/* Return the whole CPU-read allocation to the device domain. */
		codec_mm_dma_flush(dst, size, DMA_BIDIRECTIONAL);
		if (!equal) {
			d->ge2d_failed = true;
			pr_warn(DRV_NAME ": GE2D carrier byte validation failed at copy %u; using CPU until buffers reset\n",
				d->ge2d_copies);
			return false;
		}
		d->tvled_ge2d_golden_checks++;
		if (!d->ge2d_verified)
			pr_info(DRV_NAME ": GE2D carrier first frame matched CPU bytewise golden (%ux%u)\n",
				w, h);
		d->ge2d_verified = true;
	}
	s->linear_ge2d_prepared = true;
	return true;
}

/* GPU buffers remain RGBA8 for EGL import; the low three bytes hold Y:C.
 * Compact them into the slot's aligned 24-bit scanout, leaving the GPU free
 * to reuse out[buf] after this copy and retaining all 12 bits of each word.
 * A carrier uses logical RGBA bytes from the GPU; default S5 RGB888 scanout
 * is B,G,R. tvled_order permits calibration of the downstream channel map.
 */
static bool publish_linear_frame(struct dvfel_dev *d, struct dvfel_slot *s,
				 struct dvfel_cbuf *src,
				 struct vframe_s *orig, u32 w, u32 h, bool carrier)
{
	static const u8 orders[6][3] = {
		{ 2, 1, 0 }, { 0, 2, 1 }, { 0, 1, 2 },
		{ 1, 0, 2 }, { 1, 2, 0 }, { 2, 0, 1 },
	};
	const u8 *cpu;
	u8 *dst;
	u32 s0 = 0, s1 = 8, s2 = 16;
	u32 y;

	if (carrier) {
		int order = READ_ONCE(tvled_order);

		s0 = orders[order][0] * 8;
		s1 = orders[order][1] * 8;
		s2 = orders[order][2] * 8;
	}
	if (!src->mapped || !linear12_slot(d, s, w, h))
		return false;
	if (carrier && READ_ONCE(tvled_ge2d)) {
		if (s0 == 16 && s1 == 8 && s2 == 0 &&
		    publish_ge2d_carrier(d, s, src, w, h)) {
			d->tvled_ge2d_frames++;
			build_tvled_vf(s, orig, w, h);
			return true;
		}
		d->tvled_ge2d_fallback++;
	}
	if (dma_buf_begin_cpu_access(src->db, DMA_FROM_DEVICE))
		return false;
	cpu = src->map.vaddr;
	dst = codec_mm_phys_to_virt(s->linear_phys);
	if (!cpu || !dst) {
		dma_buf_end_cpu_access(src->db, DMA_FROM_DEVICE);
		return false;
	}
	/* This full CPU write invalidates the GE2D-only access assumption. */
	s->linear_ge2d_prepared = false;
	for (y = 0; y < h; y++) {
		const u8 *in = cpu + (size_t)y * w * 4;
		u8 *out = dst + (size_t)y * LINEAR12_STRIDE(w);

		compact_rgba24_row(out, in, w, s0, s1, s2);
		memset(out + w * 3, 0, LINEAR12_STRIDE(w) - w * 3);
		if ((y & 63) == 63)
			cond_resched();
	}
	codec_mm_dma_flush(dst, LINEAR12_SIZE(w, h), DMA_TO_DEVICE);
	dma_buf_end_cpu_access(src->db, DMA_FROM_DEVICE);
	if (carrier)
		build_tvled_vf(s, orig, w, h);
	else
		build_linear12_vf(s, orig, w, h);
	return true;
}

static void stat_b(struct dvfel_dev *d, s64 ub)
{
	d->us_b_sum += ub;
	d->us_b_max = max(d->us_b_max, ub);
	d->us_b_cnt++;
}

/* ------------------------------------------------------------------ */
/* debug capture                                                      */
/* ------------------------------------------------------------------ */

static int capture_copy(struct dvfel_dev *d, ulong phys, int idx, u32 w, u32 h)
{
	size_t frame = (size_t)LIN_STRIDE(w) * h;
	void *va = codec_mm_phys_to_virt(phys);

	if (!va)
		return -EFAULT;
	if (!d->cap_buf || d->cap_size < frame * 2) {
		vfree(d->cap_buf);
		d->cap_buf = vmalloc(frame * 2);
		d->cap_size = d->cap_buf ? frame * 2 : 0;
		if (!d->cap_buf)
			return -ENOMEM;
	}
	codec_mm_dma_flush(va, frame, DMA_FROM_DEVICE);
	memcpy(d->cap_buf + frame * idx, va, frame);
	return 0;
}

/*
 * frame 0: what was sent to (B) (decoder frame or compositor output),
 * frame 1: (A) of our (B) output, i.e. what the display shows
 */
static void capture_frame(struct dvfel_dev *d, struct vframe_s *vf, ulong src,
			  struct dvfel_slot *s, u32 w, u32 h)
{
	d->cap_frames = 0;
	if (capture_copy(d, src, 0, w, h))
		return;
	d->cap_frames = 1;
	d->cap_pts_us = vf->pts_us64;
	d->cap_w = w;
	d->cap_h = h;
	if (d->bufs_ok &&
	    !vicp_decompress(&d->ctx_b, &s->out_vf, d->lin_phys, w, h) &&
	    !capture_copy(d, d->lin_phys, 1, w, h))
		d->cap_frames = 2;
	pr_info(DRV_NAME ": captured pts_us %llu (%ux%u), %d frame(s)\n",
		vf->pts_us64, w, h, d->cap_frames);
}

/* ------------------------------------------------------------------ */
/* stage A: decoder frames -> (A) -> job or direct round trip         */
/* ------------------------------------------------------------------ */

static bool ensure_bufs(struct dvfel_dev *d, u32 w, u32 h)
{
	bool ok;

	mutex_lock(&d->buf_lock);
	if (d->bufs_ok && (d->buf_w != w || d->buf_h != h)) {
		unsigned long flags;
		bool idle;

		spin_lock_irqsave(&d->lock, flags);
		idle = !d->outq_cnt && !d->pend_cnt && !slots_shown_locked(d);
		spin_unlock_irqrestore(&d->lock, flags);
		if (idle)
			dvfel_free_bufs(d);
	}
	ok = d->bufs_ok ? (d->buf_w == w && d->buf_h == h) : !dvfel_alloc_bufs(d, w, h);
	if (ok && tvled_pool && READ_ONCE(tvled) && !READ_ONCE(tvled_order))
		ok = pool_prepare(d, w, h);
	mutex_unlock(&d->buf_lock);
	return ok;
}

/* phase 1 path: (A) and (B) through the internal linear buffer */
static bool test422_frame(struct dvfel_dev *d, struct dvfel_slot *s,
			  struct vframe_s *orig, u32 w, u32 h)
{
	u8 *va;
	u32 i, j;
	int pattern = READ_ONCE(test422);

	if (!linear12_slot(d, s, w, h))
		return false;
	va = codec_mm_phys_to_virt(s->linear_phys);
	if (!va)
		return false;
	/* Fill this slot every time: no shared scratch buffer or stale cache. */
	s->linear_ge2d_prepared = false;
	for (j = 0; j < h; j++) {
		u8 *row = va + (size_t)j * LINEAR12_STRIDE(w);

		for (i = 0; i < w; i++) {
			u32 y = 0x5a7, c = (i & 1) ? 0x3d5 : 0x8c3;
			u32 px;
			u8 *p = row + i * 3;

			if (pattern == 6 || pattern == 7) {
				p[0] = pattern == 6 ? 0xda : 0x80; /* B */
				p[1] = pattern == 6 ? 0xa7 : 0x80; /* G */
				p[2] = pattern == 6 ? 0x35 : 0x34 + min(i * 4 / w, 3U); /* R */
				continue;
			}
			if (pattern == 3) {
				/* Same upper ten Y bits, but low bits 0, 1, 2, 3. */
				y = 0x5a4 + min(i * 4 / w, 3U);
				c = 0x800;
			} else if (pattern == 4 || pattern == 5) {
				y = c = pattern == 4 ? 0x555 : 0xaaa;
			}
			px = pattern == 2 ? (c << 12) | y : (y << 12) | c;
			p[0] = px;
			p[1] = px >> 8;
			p[2] = px >> 16;
		}
		memset(row + w * 3, 0, LINEAR12_STRIDE(w) - w * 3);
		if ((j & 63) == 63)
			cond_resched();
	}
	codec_mm_dma_flush(va, LINEAR12_SIZE(w, h), DMA_TO_DEVICE);
	if (pattern == 6 || pattern == 7)
		build_tvled_vf(s, orig, w, h);
	else
		build_linear12_vf(s, orig, w, h);
	return true;
}

static bool process_direct(struct dvfel_dev *d, struct dvfel_slot *s,
			   struct vframe_s *vf, bool capture)
{
	u32 w = vf->compWidth, h = vf->compHeight;
	ktime_t t0, t1, t2;
	bool ok;

	if (test422) {
		mutex_lock(&d->buf_lock);
		ok = test422_frame(d, s, vf, w, h);
		mutex_unlock(&d->buf_lock);
		return ok;
	}

	mutex_lock(&d->buf_lock);
	t0 = ktime_get();
	ok = vicp_ok(d, vicp_decompress(&d->ctx_a, vf, d->lin_phys, w, h),
		     ktime_us_delta(ktime_get(), t0), "A");
	t1 = ktime_get();
	if (ok)
		ok = vicp_ok(d, vicp_compress(&d->ctx_a, d->lin_phys, s, w, h),
			     ktime_us_delta(ktime_get(), t1), "B");
	t2 = ktime_get();
	if (ok) {
		s64 ua = ktime_us_delta(t1, t0);

		d->us_a_sum += ua;
		d->us_a_max = max(d->us_a_max, ua);
		d->us_cnt++;
		stat_b(d, ktime_us_delta(t2, t1));
		build_out_vf(s, vf, w, h);
		if (capture)
			capture_frame(d, vf, d->lin_phys, s, w, h);
	}
	mutex_unlock(&d->buf_lock);
	return ok;
}

/* ------------------------------------------------------------------ */
/* hardware decoded EL (Dolby Vision dual layer decoding)             */
/* ------------------------------------------------------------------ */

/*
 * The EL picture matches the client's EL buffers by its shown size; the
 * coded (compressed) size may be larger (conformance window crop).
 */
static bool el_fits(struct vframe_s *el, u32 w, u32 h)
{
	u32 ew = el->width ? el->width : el->compWidth;
	u32 eh = el->height ? el->height : el->compHeight;

	return ew == w && eh == h && el->compWidth >= w && el->compHeight >= h;
}

static void el_put(struct dvfel_dev *d, struct vframe_s *el)
{
	vf_put(el, EL_RECV_NAME);
	vf_notify_provider(EL_RECV_NAME, VFRAME_EVENT_RECEIVER_PUT, NULL);
}

/* the BL decoder marks each picture that has an EL picture (dual layer) */
static bool bl_has_el(struct vframe_s *vf)
{
	struct provider_aux_req_s req;

	memset(&req, 0, sizeof(req));
	req.vf = vf;
	vf_notify_provider(DRV_NAME, VFRAME_EVENT_RECEIVER_GET_AUX_DATA, &req);
	return req.dv_enhance_exist;
}

/*
 * EL pictures taken from the EL decoder, waiting for their BL frame. The
 * decoders of the dual layer pair label their pictures (vf->frame_index,
 * media_modules patches 9003/9006): bits 0-15 pair a BL and an EL picture,
 * bits 16-31 are the access unit the picture was coded in. The pair label
 * is the access unit, or, for an EL coded in another picture order than
 * the BL, the display order (n-th picture of each layer); the compositor
 * then takes the RPU of the EL picture's access unit. Older ones (their BL
 * frame is gone) go back; the pair stalls when the EL is not consumed.
 */
#define PAIR(fi)	((u16)(fi))
#define PAIR_AU(fi)	((fi) >> 16)

static bool el_pair_matches(u32 bl_id, u32 el_id, bool want)
{
	if (!bl_id)
		return want; /* legacy decoder without labels: FIFO */
	return PAIR(bl_id) && PAIR(bl_id) == PAIR(el_id);
}

/* A full hold queue must not discard future EL pictures to make room for
 * an earlier missing BL. That loses the next BL's EL and repeats forever.
 * Pair labels are display order modulo 16 bits; zero is unknown. Preserve
 * the legacy FIFO policy only when the decoder provides no label at all.
 */
static int el_evict_index(struct vframe_s * const *held, int count, u32 bl_id)
{
	u16 pair = PAIR(bl_id);
	int i;

	if (!bl_id)
		return 0;
	for (i = 0; i < count; i++) {
		u16 el_pair = PAIR(held[i]->frame_index);

		if (!el_pair || (pair && (s16)(el_pair - pair) < 0))
			return i;
	}
	return -1;
}

/* (d->el_lock held) remove hold[i], keeping the order */
static struct vframe_s *el_hold_remove(struct dvfel_dev *d, int i)
{
	struct vframe_s *el = d->el_hold[i];

	memmove(&d->el_hold[i], &d->el_hold[i + 1], (d->el_nhold - i - 1) * sizeof(el));
	d->el_nhold--;
	return el;
}

static struct vframe_s *el_take(struct dvfel_dev *d, struct vframe_s *bl)
{
	struct vframe_s *drop[EL_HOLD + 1], *el = NULL;
	ktime_t t0 = ktime_get();
	u32 id = bl->frame_index;
	unsigned long flags;
	int i, ndrop;
	bool want;

	if (!READ_ONCE(d->el_prov) || !READ_ONCE(d->prov_reg))
		return NULL;
	want = bl_has_el(bl);
	for (;;) {
		bool future_full = false;

		ndrop = 0;
		spin_lock_irqsave(&d->el_lock, flags);
		while (d->el_nhold < EL_HOLD && vf_peek(EL_RECV_NAME)) {
			struct vframe_s *v = vf_get(EL_RECV_NAME);

			if (!v)
				break;
			d->el_in++;
			if (debug > 2)
				pr_info(DRV_NAME ": EL in id %u (BL id %u)\n", v->frame_index, id);
			d->el_hold[d->el_nhold++] = v;
		}
		for (i = 0; i < d->el_nhold && !el; i++) {
			/* without ids (unpatched decoder) the oldest one */
			if (el_pair_matches(id, d->el_hold[i]->frame_index, want))
				el = el_hold_remove(d, i);
		}
		/* Retire stale pictures, keeping every usable future picture. */
		for (i = 0; i < d->el_nhold;) {
			if (id && (!PAIR(d->el_hold[i]->frame_index) ||
			    (PAIR(id) && (s16)(PAIR(d->el_hold[i]->frame_index) -
					    PAIR(id)) < -EL_STALE)))
				drop[ndrop++] = el_hold_remove(d, i);
			else
				i++;
		}
		if (!el && d->el_nhold == EL_HOLD) {
			int evict = el_evict_index(d->el_hold, d->el_nhold, id);

			if (evict >= 0)
				drop[ndrop++] = el_hold_remove(d, evict);
			else
				future_full = true;
		}
		spin_unlock_irqrestore(&d->el_lock, flags);
		for (i = 0; i < ndrop; i++) {
			d->el_dropped++;
			el_put(d, drop[i]);
		}
		if (el || !want || future_full || !READ_ONCE(d->el_prov) ||
		    !READ_ONCE(d->prov_reg) || kthread_should_stop() ||
		    ktime_ms_delta(ktime_get(), t0) >= el_wait_ms)
			break;
		usleep_range(1000, 2000);
	}
	if (!want) {
		/* not flagged, but here: use it anyway, it is this frame's */
		if (!el)
			return NULL;
	}
	d->us_el_wait_max = max(d->us_el_wait_max, ktime_us_delta(ktime_get(), t0));
	if (!el && debug > 1) {
		spin_lock_irqsave(&d->el_lock, flags);
		pr_info(DRV_NAME ": no EL for BL id %u (want %d), held %d: %u %u %u %u\n",
			id, want, d->el_nhold,
			d->el_nhold > 0 ? d->el_hold[0]->frame_index : 0,
			d->el_nhold > 1 ? d->el_hold[1]->frame_index : 0,
			d->el_nhold > 2 ? d->el_hold[2]->frame_index : 0,
			d->el_nhold > 3 ? d->el_hold[3]->frame_index : 0);
		spin_unlock_irqrestore(&d->el_lock, flags);
	}
	if (!el)
		d->el_missing++;
	else if (debug > 1)
		pr_info(DRV_NAME ": EL id %u for BL id %u pts %llu, held %d\n",
			el->frame_index, id, bl->pts_us64, d->el_nhold);
	return el;
}

/* the EL decoder goes away: give back what is held */
static void el_hold_release(struct dvfel_dev *d)
{
	struct vframe_s *held[EL_HOLD];
	unsigned long flags;
	int i, n;

	spin_lock_irqsave(&d->el_lock, flags);
	n = d->el_nhold;
	memcpy(held, d->el_hold, n * sizeof(held[0]));
	d->el_nhold = 0;
	spin_unlock_irqrestore(&d->el_lock, flags);
	for (i = 0; i < n; i++)
		el_put(d, held[i]);
}

static void el_quiesce(struct dvfel_dev *d, bool reset_pending)
{
	unsigned long flags;

	spin_lock_irqsave(&d->el_lock, flags);
	d->el_prov = false;
	d->el_reset_pending = false;
	spin_unlock_irqrestore(&d->el_lock, flags);
	wait_event(d->wq_el_idle, !READ_ONCE(d->el_users));
	el_hold_release(d);
	spin_lock_irqsave(&d->el_lock, flags);
	d->el_reset_pending = reset_pending;
	spin_unlock_irqrestore(&d->el_lock, flags);
}

static int dvfel_el_event(int type, void *data, void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	unsigned long flags;

	switch (type) {
	case VFRAME_EVENT_PROVIDER_REG:
		dvfel_dbg("EL provider REG %s\n", data ? (char *)data : "");
		if (READ_ONCE(d->stopping))
			break;
		el_quiesce(d, false);
		spin_lock_irqsave(&d->el_lock, flags);
		d->el_reset_pending = false;
		d->el_prov = !READ_ONCE(d->stopping);
		spin_unlock_irqrestore(&d->el_lock, flags);
		break;
	case VFRAME_EVENT_PROVIDER_UNREG:
	case VFRAME_EVENT_PROVIDER_LIGHT_UNREG:
		dvfel_dbg("EL provider UNREG\n");
		el_quiesce(d, false);
		break;
	case VFRAME_EVENT_PROVIDER_RESET:
		el_quiesce(d, true);
		break;
	case VFRAME_EVENT_PROVIDER_VFRAME_READY:
		spin_lock_irqsave(&d->el_lock, flags);
		if (d->el_reset_pending && !READ_ONCE(d->stopping)) {
			d->el_reset_pending = false;
			d->el_prov = true;
		}
		spin_unlock_irqrestore(&d->el_lock, flags);
		break;
	case VFRAME_EVENT_PROVIDER_QUREY_STATE:
		return RECEIVER_ACTIVE;
	default:
		break;
	}
	return 0;
}

static const struct vframe_receiver_op_s dvfel_el_recv_ops = {
	.event_cb = dvfel_el_event,
};

static int dvfel_thread_a(void *data)
{
	struct dvfel_dev *d = data;

	while (!kthread_should_stop()) {
		wait_event_interruptible(d->wq_a, d->kick_a || kthread_should_stop());
		d->kick_a = false;

		for (;;) {
			struct dvfel_slot *s = NULL;
			struct dvfel_pass *p;
			struct vframe_s *vf, *el;
			unsigned long flags;
			bool gpu = false, ok = false, capture = false, el_ok = false, took_el;
			bool el_ref = false;
			u32 el_fi = 0;
			int j = -1;
			u32 gen, w, h;

			spin_lock_irqsave(&d->lock, flags);
			if (!d->prov_reg || READ_ONCE(d->stopping) ||
			    d->pend_cnt >= QUEUE_LEN || !pass_free(d)) {
				spin_unlock_irqrestore(&d->lock, flags);
				break;
			}
			if (mode) {
				/*
				 * backpressure: never fall back to passthrough, except
				 * to break the held slot deadlock (held_deadlock_locked)
				 */
				s = slot_free(d);
				if (mode == 2 && d->client_ready) {
					j = job_free(d);
					gpu = true;
				}
				if (!s && held_deadlock_locked(d)) {
					j = -1;
					gpu = false;
					d->held_breaks++;
					dvfel_dbg("all slots held after a reset: pass a frame through\n");
				} else if (!s || (gpu && j < 0)) {
					d->slot_waits++;
					spin_unlock_irqrestore(&d->lock, flags);
					break;
				} else {
					s->state = SLOT_BUSY;
					s->tvled_carrier = false;
					if (gpu)
						d->jobs[j].state = JOB_RESERVED;
				}
			}
			d->bl_users++;
			gen = d->gen;
			spin_unlock_irqrestore(&d->lock, flags);

			vf = vf_peek(DRV_NAME) ? vf_get(DRV_NAME) : NULL;
			if (!vf) {
				spin_lock_irqsave(&d->lock, flags);
				if (s)
					s->state = SLOT_FREE;
				if (j >= 0)
					d->jobs[j].state = JOB_FREE;
				spin_unlock_irqrestore(&d->lock, flags);
				bl_ref_put(d);
				break;
			}
			d->frames_in++;
			w = vf->compWidth;
			h = vf->compHeight;
			el = NULL;
			took_el = false;

			if (s && frame_supported(vf) && ensure_bufs(d, w, h)) {
				capture = atomic_xchg(&d->capture_req, 0) ||
					  (capture_pts && vf->pts_us64 == capture_pts);
				if (gpu && d->client_w == w && d->client_h == h) {
					ktime_t t0 = ktime_get();
					s64 ua;

					ok = vicp_ok(d, vicp_decompress(&d->ctx_a, vf,
									d->cin[j].phys, w, h),
						     ktime_us_delta(ktime_get(), t0), "A");
					ua = ktime_us_delta(ktime_get(), t0);
					if (ok) {
						d->us_a_sum += ua;
						d->us_a_max = max(d->us_a_max, ua);
						d->us_cnt++;
					}
					/* after the BL: by now its EL picture is usually there */
					if (ok) {
						el_ref = el_ref_get(d);
						if (el_ref)
							el = el_take(d, vf);
						took_el = true;
					}
					if (ok && el && d->client_el_w && frame_supported(el) &&
					    el_fits(el, d->client_el_w, d->client_el_h)) {
						t0 = ktime_get();
						el_ok = vicp_ok(d, vicp_decompress(&d->ctx_a, el,
										   d->cel[j].phys,
										   d->client_el_w,
										   d->client_el_h),
								ktime_us_delta(ktime_get(), t0), "A EL");
						ua = ktime_us_delta(ktime_get(), t0);
						if (el_ok) {
							el_fi = el->frame_index;
							d->el_used++;
							d->us_el_sum += ua;
							d->us_el_max = max(d->us_el_max, ua);
							d->us_el_cnt++;
						}
					}
				} else {
					if (j >= 0) {
						spin_lock_irqsave(&d->lock, flags);
						d->jobs[j].state = JOB_FREE;
						spin_unlock_irqrestore(&d->lock, flags);
						j = -1;
					}
					gpu = false;
					ok = !(mode == 2 && READ_ONCE(tvled)) &&
					     process_direct(d, s, vf, capture);
				}
			} else if (j >= 0) {
				spin_lock_irqsave(&d->lock, flags);
				d->jobs[j].state = JOB_FREE;
				spin_unlock_irqrestore(&d->lock, flags);
				j = -1;
				gpu = false;
			}
			if (!took_el) {
				el_ref = el_ref_get(d);
				if (el_ref)
					el = el_take(d, vf);
			}
			if (el) {
				if (!el_ok)
					d->el_dropped++;
				el_put(d, el);
			}
			if (el_ref)
				el_ref_put(d);
			spin_lock_irqsave(&d->lock, flags);
			if (gen != d->gen) {
				/* reset/unreg while processing: drop the frame */
				if (s)
					s->state = SLOT_FREE;
				if (j >= 0)
					d->jobs[j].state = JOB_FREE;
				spin_unlock_irqrestore(&d->lock, flags);
				bl_ref_put(d);
				continue;
			}
			if (ok && gpu) {
				struct dvfel_kjob *jb = &d->jobs[j];

				jb->id = ++d->job_seq;
				jb->gen = gen;
				jb->pts_us = vf->pts_us64;
				jb->bl_au = PAIR_AU(vf->frame_index);
				jb->el_au = el_ok ? PAIR_AU(el_fi) : jb->bl_au;
				jb->flags = (d->new_stream ? DVFEL_JOB_NEW_STREAM : 0) |
					    (el_ok ? DVFEL_JOB_EL : 0) |
					    (READ_ONCE(tvled) ? tvled_job_flags_locked(d) :
					     ((d->cout[j].mapped && READ_ONCE(output12)) ?
					      DVFEL_JOB_LINEAR12 : 0));
				jb->slot = s;
				jb->pool_token = 0;
				if ((jb->flags & DVFEL_JOB_TVLED) && tvled_pool &&
				    !READ_ONCE(tvled_order) && d->pool_ready &&
				    d->nslots <= DVFEL_POOL_MAX_SLOTS && s->pool_mem) {
					unsigned int index = s - d->slots;

					jb->pool_token = d->pool_token;
					jb->flags |= DVFEL_JOB_TVLED_POOL |
						(index << DVFEL_JOB_POOL_SLOT_SHIFT) |
						((u32)(d->pool_token & 0xffff) << DVFEL_JOB_POOL_COOKIE_SHIFT);
					s->pool_writer = jb->id;
				}
				jb->orig = vf;
				jb->capture = capture;
				jb->posted = ktime_get();
				jb->state = JOB_POSTED;
				d->new_stream = false;
				d->gpu_jobs++;
				pend_push(d, NULL, j, gen);
				wake_up_interruptible(&d->wq_client);
			} else if (ok) {
				s->orig = vf;
				s->gen = gen;
				s->state = SLOT_OUT;
				s->taken = false;
				pend_push(d, &s->out_vf, -1, gen);
				d->frames_proc++;
			} else {
				if (s)
					s->state = SLOT_FREE;
				if (j >= 0)
					d->jobs[j].state = JOB_FREE;
				if (mode == 2 && READ_ONCE(tvled)) {
					d->tvled_dropped++;
					spin_unlock_irqrestore(&d->lock, flags);
					vf_put(vf, DRV_NAME);
					bl_ref_put(d);
					continue;
				}
				p = pass_free(d);
				p->vf = vf;
				p->gen = gen;
				pend_push(d, vf, -1, gen);
				d->frames_pass++;
			}
			spin_unlock_irqrestore(&d->lock, flags);
			bl_ref_put(d);
			d->kick_b = true;
			wake_up_interruptible(&d->wq_b);
		}
	}
	return 0;
}

/* ------------------------------------------------------------------ */
/* stage B: pending queue (display order) -> (B) -> display           */
/* ------------------------------------------------------------------ */

enum pend_ready { PEND_NONE, PEND_READY, PEND_WAIT };

static enum pend_ready pend_head_ready(struct dvfel_dev *d, ktime_t *deadline)
{
	struct dvfel_pend *p;
	struct dvfel_kjob *j;

	if (!d->pend_cnt)
		return PEND_NONE;
	p = &d->pendq[d->pend_head];
	if (p->job < 0 || p->gen != d->gen)
		return PEND_READY;
	j = &d->jobs[p->job];
	if (j->state == JOB_DONE || !d->client_ready)
		return PEND_READY;
	*deadline = ktime_add_ms(j->posted, gpu_timeout_ms);
	return ktime_after(ktime_get(), *deadline) ? PEND_READY : PEND_WAIT;
}

static void finish_job(struct dvfel_dev *d, struct dvfel_pend *pe)
{
	struct dvfel_kjob *j = &d->jobs[pe->job];
	struct dvfel_slot *s = j->slot;
	struct vframe_s *orig = j->orig;
	u32 w = orig->compWidth, h = orig->compHeight;
	bool composed, linear12, carrier, pool, capture, ok;
	unsigned long flags;
	ulong src;
	ktime_t t0;
	s64 lat, tvled_us = -1;

	spin_lock_irqsave(&d->lock, flags);
	pool = j->state == JOB_DONE && j->status == DVFEL_DONE_TVLED_POOL &&
	       pool_job_valid_locked(d, j);
	carrier = pool || (j->state == JOB_DONE &&
		  (j->flags & DVFEL_JOB_TVLED) && j->status == DVFEL_DONE_TVLED);
	linear12 = j->state == JOB_DONE &&
		   (j->flags & DVFEL_JOB_LINEAR12) && j->status == DVFEL_DONE_LINEAR12;
	composed = carrier || linear12 || (j->state == JOB_DONE &&
		   !(j->flags & (DVFEL_JOB_LINEAR12 | DVFEL_JOB_TVLED)) &&
		   j->status == DVFEL_DONE_COMPOSED);
	if (j->state == JOB_DONE) {
		lat = ktime_us_delta(ktime_get(), j->posted);
		d->us_gpu_sum += lat;
		d->us_gpu_max = max(d->us_gpu_max, lat);
		d->us_gpu_cnt++;
		j->state = JOB_FINISH;
	} else {
		/* timed out or compositor gone: show the base layer */
		d->gpu_timeouts++;
		if (j->state == JOB_POSTED)
			j->state = JOB_FINISH;
		/* JOB_TAKEN stays: out[] may still be written, in[] is read only */
	}
	src = composed ? d->cout[pe->job].phys : d->cin[pe->job].phys;
	spin_unlock_irqrestore(&d->lock, flags);

	capture = j->capture;
	j->capture = false;
	mutex_lock(&d->buf_lock);
	ok = false;
	if (linear12 || carrier) {
		t0 = ktime_get();
		if (pool)
			ok = publish_pool_frame(d, j, orig, w, h);
		else
			ok = publish_linear_frame(d, s, &d->cout[pe->job], orig, w, h, carrier);
		if (carrier)
			tvled_us = ktime_us_delta(ktime_get(), t0);
		if (!ok) {
			/* Packed 12-bit words cannot go through the 10-bit VICP input. */
			src = d->cin[pe->job].phys;
			composed = false;
		}
	}
	/* A TV-led transport must never be mixed with ordinary AFBC frames.
	 * On missing metadata, timeout, or publication failure retain the last
	 * legal carrier on screen by returning this decoder frame upstream.
	 */
	if (!ok && !(j->flags & DVFEL_JOB_TVLED)) {
		t0 = ktime_get();
		ok = vicp_ok(d, vicp_compress(&d->ctx_b, src, s, w, h),
			     ktime_us_delta(ktime_get(), t0), "B");
		if (ok) {
			stat_b(d, ktime_us_delta(ktime_get(), t0));
			build_out_vf(s, orig, w, h);
			if (capture)
				capture_frame(d, orig, src, s, w, h);
		}
	}
	mutex_unlock(&d->buf_lock);
	if (carrier && ok && !READ_ONCE(d->display_lifecycle) &&
	    !tvled_signal_acquire(d, pe->gen, false)) {
		ok = false;
		composed = false;
	}

	spin_lock_irqsave(&d->lock, flags);
	if (tvled_us >= 0) {
		d->us_tvled_sum += tvled_us;
		d->us_tvled_max = max(d->us_tvled_max, tvled_us);
		d->us_tvled_cnt++;
	}
	if (composed)
		d->gpu_composed++;
	else
		d->gpu_fallback++;
	if (j->flags & DVFEL_JOB_TVLED) {
		if (carrier && ok && composed)
			d->tvled_frames++;
		else
			d->tvled_dropped++;
	}
	if (j->flags & DVFEL_JOB_LINEAR12) {
		if (linear12 && ok && composed)
			d->linear12_frames++;
		else
			d->linear12_fallback++;
	}
	if (pe->gen != d->gen) {
		s->state = SLOT_FREE;
	} else if (ok) {
		s->orig = orig;
		s->gen = pe->gen;
		s->tvled_carrier = carrier;
		s->state = SLOT_OUT;
		s->taken = false;
		outq_push(d, &s->out_vf);
		d->frames_proc++;
		if (pool)
			d->tvled_pool_frames++;
	} else {
		/* cannot show it: give the decoder frame back */
		s->state = SLOT_FREE;
		spin_unlock_irqrestore(&d->lock, flags);
		vf_put(orig, DRV_NAME);
		spin_lock_irqsave(&d->lock, flags);
	}
	if (j->state == JOB_FINISH)
		j->state = JOB_DONE;	/* so job_release frees it */
	job_release(d, j);
	spin_unlock_irqrestore(&d->lock, flags);
}

static int dvfel_thread_b(void *data)
{
	struct dvfel_dev *d = data;

	while (!kthread_should_stop()) {
		struct dvfel_pend pe;
		unsigned long flags;
		enum pend_ready r;
		ktime_t deadline = 0;

		spin_lock_irqsave(&d->lock, flags);
		r = pend_head_ready(d, &deadline);
		spin_unlock_irqrestore(&d->lock, flags);

		if (r == PEND_NONE) {
			wait_event_interruptible(d->wq_b, d->kick_b || kthread_should_stop());
			d->kick_b = false;
			continue;
		}
		if (r == PEND_WAIT) {
			s64 left = ktime_us_delta(deadline, ktime_get());

			wait_event_interruptible_timeout(d->wq_b, d->kick_b || kthread_should_stop(),
							 usecs_to_jiffies(max_t(s64, left, 1000)));
			d->kick_b = false;
			continue;
		}

		spin_lock_irqsave(&d->lock, flags);
		pe = d->pendq[d->pend_head];
		pend_pop(d);
		if (pe.gen != d->gen || !bl_ref_get_locked(d)) {
			/*
			 * stale entry from before a reset: the reset already
			 * freed SLOT_OUT slots of direct entries (which may be
			 * reused by now); a job still owns its BUSY slot
			 */
			if (pe.job >= 0) {
				struct dvfel_kjob *j = &d->jobs[pe.job];

				if (j->slot)
					j->slot->state = SLOT_FREE;
				if (j->state == JOB_POSTED)
					j->state = JOB_DONE;
				job_release(d, j);
			}
			spin_unlock_irqrestore(&d->lock, flags);
			d->kick_a = true;
			wake_up_interruptible(&d->wq_a);
			continue;
		}
		if (pe.job < 0) {
			outq_push(d, pe.vf);
			spin_unlock_irqrestore(&d->lock, flags);
		} else {
			spin_unlock_irqrestore(&d->lock, flags);
			finish_job(d, &pe);
		}
		bl_ref_put(d);
		vf_notify_receiver(DRV_NAME, VFRAME_EVENT_PROVIDER_VFRAME_READY, NULL);
		d->kick_a = true;
		wake_up_interruptible(&d->wq_a);
	}
	return 0;
}

static void dvfel_kick(struct dvfel_dev *d)
{
	d->kick_a = true;
	wake_up_interruptible(&d->wq_a);
}

/* ------------------------------------------------------------------ */
/* provider side (downstream calls us, often from vsync irq)          */
/* ------------------------------------------------------------------ */

static struct vframe_s *dvfel_peek(void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	struct vframe_s *vf = NULL;
	unsigned long flags;

	spin_lock_irqsave(&d->lock, flags);
	if (d->outq_cnt)
		vf = d->outq[d->outq_head];
	spin_unlock_irqrestore(&d->lock, flags);
	return vf;
}

static struct vframe_s *dvfel_get(void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	struct vframe_s *vf;
	unsigned long flags;

	spin_lock_irqsave(&d->lock, flags);
	vf = outq_pop(d);
	if (vf) {
		struct dvfel_slot *s = slot_of(d, vf);

		if (s) {
			s->taken = true;
			tvled_delivered_locked(d, s, vf);
		}
		d->shown_new = true;
	}
	spin_unlock_irqrestore(&d->lock, flags);
	if (vf)
		dvfel_kick(d);
	return vf;
}

static void dvfel_put(struct vframe_s *vf, void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	struct vframe_s *upstream = NULL;
	struct dvfel_slot *s;
	unsigned long flags;
	int i;

	if (!vf)
		return;
	spin_lock_irqsave(&d->lock, flags);
	s = slot_of(d, vf);
	if (s) {
		if (s->state == SLOT_OUT && s->gen == d->gen)
			upstream = s->orig;
		/*
		 * a held slot waits for the keeper (slot_released_locked()),
		 * unless the display returned it and the keeper does not hold it
		 */
		if (s->state == SLOT_OUT || (s->state == SLOT_HELD && !slot_kept(s)))
			s->state = SLOT_FREE;
		s->orig = NULL;
	} else {
		for (i = 0; i < QUEUE_LEN; i++) {
			if (d->pass[i].vf == vf) {
				if (d->pass[i].gen == d->gen)
					upstream = vf;
				d->pass[i].vf = NULL;
				break;
			}
		}
	}
	if (upstream && !bl_ref_get_locked(d))
		upstream = NULL;
	spin_unlock_irqrestore(&d->lock, flags);

	if (upstream) {
		vf_put(upstream, DRV_NAME);
		bl_ref_put(d);
	}
	dvfel_kick(d);
}

/* events sent upstream by the display side: map our copy back */
static int dvfel_prov_event(int type, void *data, void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	struct vframe_s **pvf = NULL, *mine = NULL, *upstream = NULL;
	unsigned long flags;
	bool valid = true, held = false;
	int i;

	if (debug > 1 || (debug && !(type & VFRAME_EVENT_RECEIVER_GET_AUX_DATA)))
		dvfel_dbg("downstream event 0x%x\n", type);
	if (data && (type & (VFRAME_EVENT_RECEIVER_GET_AUX_DATA |
			     VFRAME_EVENT_RECEIVER_DISP_MODE |
			     VFRAME_EVENT_RECEIVER_REQ_STATE))) {
		/* all three request structs start with 'struct vframe_s *vf' */
		pvf = data;
		mine = *pvf;
	}
	/* This callback may run from vsync IRQ: take only the spinlocked gate,
	 * never a sleeping mutex. UNREG waits for this forwarder before the
	 * decoder can destroy orig or its metadata lookup state.
	 */
	spin_lock_irqsave(&d->lock, flags);
	if (pvf && mine) {
		struct dvfel_slot *s = slot_of(d, mine);

		if (s) {
			valid = s->orig && s->gen == d->gen;
			upstream = s->orig;
		} else {
			valid = false;
			for (i = 0; i < QUEUE_LEN; i++)
				if (d->pass[i].vf == mine && d->pass[i].gen == d->gen) {
					valid = true;
					break;
				}
		}
	}
	if (valid)
		held = bl_ref_get_locked(d);
	spin_unlock_irqrestore(&d->lock, flags);
	if (!held)
		return 0;
	if (pvf && upstream)
		*pvf = upstream;
	vf_notify_provider(DRV_NAME, type, data);
	if (pvf)
		*pvf = mine;
	bl_ref_put(d);
	return 0;
}

static int dvfel_vf_states(struct vframe_states *states, void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	unsigned long flags;
	int i, nfree = 0;

	spin_lock_irqsave(&d->lock, flags);
	for (i = 0; i < d->nslots; i++)
		if (d->slots[i].state == SLOT_FREE)
			nfree++;
	states->vf_pool_size = d->nslots;
	states->buf_free_num = nfree;
	states->buf_recycle_num = 0;
	states->buf_avail_num = d->outq_cnt;
	spin_unlock_irqrestore(&d->lock, flags);
	return 0;
}

static const struct vframe_operations_s dvfel_vf_ops = {
	.peek = dvfel_peek,
	.get = dvfel_get,
	.put = dvfel_put,
	.event_cb = dvfel_prov_event,
	.vf_states = dvfel_vf_states,
};

/* ------------------------------------------------------------------ */
/* receiver side (upstream decoder events)                            */
/* ------------------------------------------------------------------ */

static int dvfel_recv_event(int type, void *data, void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	unsigned long flags;

	if (type != VFRAME_EVENT_PROVIDER_VFRAME_READY)
		dvfel_dbg("upstream event 0x%x\n", type);
	switch (type) {
	case VFRAME_EVENT_PROVIDER_REG:
		dvfel_dbg("upstream REG %s\n", data ? (char *)data : "");
		if (READ_ONCE(d->stopping))
			break;
		cancel_delayed_work_sync(&d->free_work);
		bl_quiesce(d, false);
		spin_lock_irqsave(&d->lock, flags);
		d->prov_reg = !READ_ONCE(d->stopping);
		spin_unlock_irqrestore(&d->lock, flags);
		if (READ_ONCE(d->stopping))
			break;
		vf_reg_provider(&d->prov);
		dvfel_kick(d);
		break;
	case VFRAME_EVENT_PROVIDER_UNREG:
		dvfel_dbg("upstream UNREG\n");
		bl_quiesce(d, false);
		if (!READ_ONCE(d->display_lifecycle))
			tvled_signal_release(d);
		vf_unreg_provider(&d->prov);
		d->kick_b = true;
		wake_up_interruptible(&d->wq_b);
		/* the display may still scan out our last buffer for a while */
		schedule_delayed_work(&d->free_work, msecs_to_jiffies(3000));
		break;
	case VFRAME_EVENT_PROVIDER_LIGHT_UNREG:
		bl_quiesce(d, false);
		vf_light_unreg_provider(&d->prov);
		d->kick_b = true;
		wake_up_interruptible(&d->wq_b);
		break;
	case VFRAME_EVENT_PROVIDER_RESET:
		bl_quiesce(d, true);
		vf_notify_receiver(DRV_NAME, type, data);
		d->kick_b = true;
		wake_up_interruptible(&d->wq_b);
		break;
	case VFRAME_EVENT_PROVIDER_VFRAME_READY:
		spin_lock_irqsave(&d->lock, flags);
		if (d->bl_reset_pending && !READ_ONCE(d->stopping)) {
			d->bl_reset_pending = false;
			d->prov_reg = true;
		}
		spin_unlock_irqrestore(&d->lock, flags);
		dvfel_kick(d);
		break;
	case VFRAME_EVENT_PROVIDER_QUREY_STATE:
		return RECEIVER_ACTIVE;
	default:
		/* START, FR_HINT, FR_END_HINT, ... */
		return vf_notify_receiver(DRV_NAME, type, data);
	}
	return 0;
}

static const struct vframe_receiver_op_s dvfel_recv_ops = {
	.event_cb = dvfel_recv_event,
};

/* ------------------------------------------------------------------ */
/* /dev/dvfel (compositor)                                            */
/* ------------------------------------------------------------------ */

static void cbuf_put(struct dvfel_cbuf *b)
{
	if (b->mapped)
		dma_buf_vunmap(b->db, &b->map);
	if (b->sgt)
		dma_buf_unmap_attachment(b->att, b->sgt, DMA_BIDIRECTIONAL);
	if (b->att)
		dma_buf_detach(b->db, b->att);
	if (b->db)
		dma_buf_put(b->db);
	memset(b, 0, sizeof(*b));
}

/* import a physically contiguous dma-buf of at least @size bytes */
static int cbuf_get(struct dvfel_dev *d, struct dvfel_cbuf *b, int fd, size_t size, bool cpu_map)
{
	struct scatterlist *sg;
	phys_addr_t next;
	size_t len = 0;
	int i;

	b->db = dma_buf_get(fd);
	if (IS_ERR(b->db)) {
		b->db = NULL;
		return -EBADF;
	}
	/* Pool outputs, including retired generations, can only be GPU outputs.
	 * Re-importing one as VICP input/cout could overwrite a scanned frame.
	 */
	if (b->db->ops == &pool_dma_ops)
		goto fail;
	b->att = dma_buf_attach(b->db, d->misc.this_device);
	if (IS_ERR(b->att)) {
		b->att = NULL;
		goto fail;
	}
	b->sgt = dma_buf_map_attachment(b->att, DMA_BIDIRECTIONAL);
	if (IS_ERR(b->sgt)) {
		b->sgt = NULL;
		goto fail;
	}
	b->phys = page_to_phys(sg_page(b->sgt->sgl)) + b->sgt->sgl->offset;
	next = b->phys;
	for_each_sgtable_sg(b->sgt, sg, i) {
		if (page_to_phys(sg_page(sg)) + sg->offset != next)
			goto fail;	/* not contiguous */
		next += sg->length;
		len += sg->length;
	}
	if (len < size)
		goto fail;
	/* Only output buffers need CPU access; mapping is optional for 10-bit. */
	if (cpu_map && !dma_buf_vmap(b->db, &b->map)) {
		if (dma_buf_map_is_set(&b->map) && !b->map.is_iomem)
			b->mapped = true;
		else
			dma_buf_vunmap(b->db, &b->map);
	}
	return 0;
fail:
	cbuf_put(b);
	return -EINVAL;
}

static bool client_jobs_busy(struct dvfel_dev *d)
{
	unsigned long flags;
	bool busy = false;
	int i;

	spin_lock_irqsave(&d->lock, flags);
	for (i = 0; i < d->client_nbufs; i++)
		if (d->jobs[i].state != JOB_FREE)
			busy = true;
	spin_unlock_irqrestore(&d->lock, flags);
	return busy;
}

/* d->client_lock held */
static void client_unreg(struct dvfel_dev *d)
{
	unsigned long flags;
	int i;

	spin_lock_irqsave(&d->lock, flags);
	d->client_ready = false;
	d->client_pool_token = 0;
	for (i = 0; i < d->client_nbufs; i++)
		if (d->jobs[i].state == JOB_TAKEN) {
			struct dvfel_kjob *j = &d->jobs[i];
			struct dvfel_slot *s = j->slot;

			/* Closing the producer is not a GPU-completion fence. Retire
			 * its writable allocation; lingering CL imports keep the old
			 * metadata/mm alive and can never touch a new pool allocation.
			 */
			if ((j->flags & DVFEL_JOB_TVLED_POOL) && s && s->pool_mem &&
			    s->pool_writer == j->id)
				s->pool_mem->abandoned = true;
			pool_writer_done_locked(&d->jobs[i]);
			if (d->jobs[i].orphan)
				d->jobs[i].slot = NULL;
			/* nobody will report it any more */
			d->jobs[i].state = d->jobs[i].orphan ? JOB_FREE : JOB_DONE;
			d->jobs[i].status = DVFEL_DONE_PASSTHROUGH;
			d->jobs[i].orphan = false;
		}
	spin_unlock_irqrestore(&d->lock, flags);
	if (!READ_ONCE(d->display_lifecycle))
		tvled_signal_release(d);
	d->kick_b = true;
	wake_up_interruptible(&d->wq_b);
	wake_up_interruptible(&d->wq_client);

	/* A may still write cin/cel and B may still read cout (GE2D has a
	 * one-second hardware timeout). A deadline can warn, but cannot make
	 * releasing live DMA buffers safe. ready=false prevents new reserves
	 * and lets pending jobs drain without another userspace completion.
	 */
	i = 0;
	while (client_jobs_busy(d)) {
		msleep(10);
		if (++i == 100)
			pr_warn(DRV_NAME ": waiting for compositor DMA buffers to drain at unregister\n");
	}

	mutex_lock(&d->buf_lock);
	for (i = 0; i < d->nslots; i++) {
		struct dvfel_slot *s = &d->slots[i];
		struct dvfel_pool_mem *p;
		struct codec_mm_s *mm;

		spin_lock_irqsave(&d->lock, flags);
		p = s->pool_mem;
		if (!p || !p->abandoned || WARN_ON(s->state == SLOT_OUT || s->state == SLOT_HELD)) {
			spin_unlock_irqrestore(&d->lock, flags);
			continue;
		}
		mm = s->linear_mm;
		s->linear_mm = NULL;
		s->pool_mem = NULL;
		s->linear_phys = 0;
		s->pool_writer = 0;
		s->linear_ge2d_prepared = false;
		s->state = SLOT_FREE;
		d->pool_ready = false;
		pool_next_token_locked(d);
		spin_unlock_irqrestore(&d->lock, flags);
		codec_mm_release(mm, DRV_NAME);
		pool_mem_put(p);
	}
	for (i = 0; i < d->client_nbufs; i++) {
		cbuf_put(&d->cin[i]);
		cbuf_put(&d->cout[i]);
		cbuf_put(&d->cel[i]);
		d->jobs[i].state = JOB_FREE;
	}
	d->client_nbufs = 0;
	d->client_el_w = d->client_el_h = 0;
	mutex_unlock(&d->buf_lock);
}

static long ioc_reg_bufs(struct dvfel_dev *d, struct file *f, void __user *arg)
{
	struct dvfel_reg_bufs r;
	size_t size;
	int i, ret;

	if (copy_from_user(&r, arg, sizeof(r)))
		return -EFAULT;
	if (!r.count || r.count > DVFEL_MAX_BUFS || !r.width || !r.height ||
	    r.width > MAX_W || r.height > MAX_H || !wmif_bg_width(r.width))
		return -EINVAL;
	if (r.el_width && (!r.el_height || r.el_width > r.width || r.el_height > r.height ||
			   (r.el_width | r.el_height) & 1 || !wmif_bg_width(r.el_width)))
		return -EINVAL;
	size = LIN_SIZE(r.width, r.height);

	mutex_lock(&d->client_lock);
	if (d->client && d->client != f) {
		mutex_unlock(&d->client_lock);
		return -EBUSY;
	}
	if (d->client_nbufs)
		client_unreg(d);
	for (i = 0; i < r.count; i++) {
		ret = cbuf_get(d, &d->cin[i], r.in_fd[i], size, false);
		if (!ret)
			ret = cbuf_get(d, &d->cout[i], r.out_fd[i], size, true);
		if (!ret && r.el_width)
			ret = cbuf_get(d, &d->cel[i], r.el_fd[i], LIN_SIZE(r.el_width, r.el_height), false);
		if (ret) {
			d->client_nbufs = i + 1;
			client_unreg(d);
			mutex_unlock(&d->client_lock);
			pr_err(DRV_NAME ": buffer %d rejected (need %zu contiguous bytes)\n",
			       i, size);
			return ret;
		}
	}
	d->client = f;
	d->client_w = r.width;
	d->client_h = r.height;
	d->client_nbufs = r.count;
	d->client_el_w = r.el_width;
	d->client_el_h = r.el_width ? r.el_height : 0;
	memset(d->jobs, 0, sizeof(d->jobs));
	d->client_ready = true;
	d->new_stream = true;
	mutex_unlock(&d->client_lock);
	pr_info(DRV_NAME ": compositor registered %u buffer pairs %ux%u, EL %ux%u\n",
		r.count, r.width, r.height, r.el_width, r.el_height);
	return 0;
}

static bool job_posted(struct dvfel_dev *d, int *idx)
{
	unsigned long flags;
	u32 best = 0;
	int i;

	*idx = -1;
	spin_lock_irqsave(&d->lock, flags);
	for (i = 0; i < d->client_nbufs; i++) {
		struct dvfel_kjob *j = &d->jobs[i];

		if (j->state == JOB_POSTED && (*idx < 0 || (s32)(j->id - best) < 0)) {
			*idx = i;
			best = j->id;
		}
	}
	spin_unlock_irqrestore(&d->lock, flags);
	return *idx >= 0 || !d->client_ready;
}

static long ioc_wait_job(struct dvfel_dev *d, struct file *f, void __user *arg)
{
	struct dvfel_job u;
	unsigned long flags;
	long ret;
	int i;

	if (copy_from_user(&u, arg, sizeof(u)))
		return -EFAULT;
	if (d->client != f || !d->client_ready)
		return -ENODEV;
	ret = wait_event_interruptible_timeout(d->wq_client, job_posted(d, &i),
					       msecs_to_jiffies(u.timeout_ms));
	if (ret < 0)
		return ret;
	spin_lock_irqsave(&d->lock, flags);
	if (i < 0 || d->jobs[i].state != JOB_POSTED || !d->client_ready) {
		spin_unlock_irqrestore(&d->lock, flags);
		return d->client_ready ? -ETIMEDOUT : -ENODEV;
	}
	d->jobs[i].state = JOB_TAKEN;
	u.id = d->jobs[i].id;
	u.buf = i;
	u.width = d->client_w;
	u.height = d->client_h;
	u.flags = d->jobs[i].flags;
	u.pts_us = d->jobs[i].pts_us;
	u.bl_au = d->jobs[i].bl_au;
	u.el_au = d->jobs[i].el_au;
	spin_unlock_irqrestore(&d->lock, flags);
	return copy_to_user(arg, &u, sizeof(u)) ? -EFAULT : 0;
}

static long ioc_job_done(struct dvfel_dev *d, struct file *f, void __user *arg)
{
	struct dvfel_job_done u;
	unsigned long flags;
	int i, ret = -ENOENT;

	if (copy_from_user(&u, arg, sizeof(u)))
		return -EFAULT;
	if (d->client != f)
		return -ENODEV;
	spin_lock_irqsave(&d->lock, flags);
	for (i = 0; i < d->client_nbufs; i++) {
		struct dvfel_kjob *j = &d->jobs[i];

		if (j->id != u.id || j->state != JOB_TAKEN)
			continue;
		pool_writer_done_locked(j);
		if (j->orphan) {
			j->state = JOB_FREE;
			j->orphan = false;
			j->slot = NULL;
		} else {
			int expected = (j->flags & DVFEL_JOB_TVLED) ? DVFEL_DONE_TVLED :
				       ((j->flags & DVFEL_JOB_LINEAR12) ?
					DVFEL_DONE_LINEAR12 : DVFEL_DONE_COMPOSED);
			bool pool_done = u.status == DVFEL_DONE_TVLED_POOL &&
				pool_job_valid_locked(d, j) &&
				d->client_pool_token == j->pool_token;

			j->state = JOB_DONE;
			if (u.status != DVFEL_DONE_PASSTHROUGH && u.status != expected && !pool_done) {
				j->status = DVFEL_DONE_PASSTHROUGH;
				ret = -EINVAL;
				break;
			}
			j->status = u.status;
		}
		ret = 0;
		break;
	}
	spin_unlock_irqrestore(&d->lock, flags);
	d->kick_b = true;
	wake_up_interruptible(&d->wq_b);
	dvfel_kick(d);
	return ret;
}

static struct dvfel_kjob *pool_find_job_locked(struct dvfel_dev *d, u32 id)
{
	int i;

	for (i = 0; i < d->client_nbufs; i++)
		if (d->jobs[i].id == id && d->jobs[i].state == JOB_TAKEN &&
		    !d->jobs[i].orphan && pool_job_valid_locked(d, &d->jobs[i]))
			return &d->jobs[i];
	return NULL;
}

static long ioc_get_pool(struct dvfel_dev *d, struct file *f, void __user *arg)
{
	struct dvfel_pool u;
	struct dma_buf *db[DVFEL_POOL_MAX_SLOTS] = { NULL };
	struct dvfel_kjob *j;
	unsigned long flags;
	u64 cached;
	u32 id;
	bool reuse;
	unsigned int i;
	int ret = 0;

	if (copy_from_user(&u, arg, sizeof(u)))
		return -EFAULT;
	if (u.reserved)
		return -EINVAL;
	id = u.job_id;
	cached = u.token;
	memset(&u, 0, sizeof(u));
	u.job_id = id;
	for (i = 0; i < DVFEL_POOL_MAX_SLOTS; i++)
		u.fd[i] = -1;
	mutex_lock(&d->client_lock);
	mutex_lock(&d->buf_lock);
	spin_lock_irqsave(&d->lock, flags);
	if (d->client != f || !d->client_ready) {
		ret = -ENODEV;
	} else if (!(j = pool_find_job_locked(d, id))) {
		ret = -ESTALE;
	} else {
		u.width = d->buf_w;
		u.height = d->buf_h;
		u.stride = LINEAR12_STRIDE(u.width);
		u.count = d->nslots;
		u.token = j->pool_token;
	}
	reuse = !ret && cached == u.token && d->client_pool_token == u.token;
	spin_unlock_irqrestore(&d->lock, flags);
	if (ret)
		goto out;
	for (i = 0; !reuse && i < u.count; i++) {
		u.fd[i] = get_unused_fd_flags(O_CLOEXEC);
		if (u.fd[i] < 0) {
			ret = u.fd[i];
			u.fd[i] = -1;
			goto rollback;
		}
		db[i] = pool_export(d->slots[i].pool_mem);
		if (IS_ERR(db[i])) {
			ret = PTR_ERR(db[i]);
			db[i] = NULL;
			goto rollback;
		}
	}
	/* Reset may advance the epoch while exporting. No FD was installed yet. */
	spin_lock_irqsave(&d->lock, flags);
	j = pool_find_job_locked(d, id);
	if (!j || j->pool_token != u.token)
		ret = -ESTALE;
	spin_unlock_irqrestore(&d->lock, flags);
	if (ret)
		goto rollback;
	if (copy_to_user(arg, &u, sizeof(u))) {
		ret = -EFAULT;
		goto rollback;
	}
	/* All fallible operations completed. FD installation consumes each
	 * exported file reference; the slot never keeps a dma-buf reference.
	 */
	for (i = 0; !reuse && i < u.count; i++) {
		fd_install(u.fd[i], db[i]->file);
		db[i] = NULL;
	}
	spin_lock_irqsave(&d->lock, flags);
	d->client_pool_token = u.token;
	spin_unlock_irqrestore(&d->lock, flags);
	goto out;
rollback:
	for (i = 0; i < DVFEL_POOL_MAX_SLOTS; i++) {
		if (db[i])
			dma_buf_put(db[i]);
		if (u.fd[i] >= 0)
			put_unused_fd(u.fd[i]);
	}
out:
	mutex_unlock(&d->buf_lock);
	mutex_unlock(&d->client_lock);
	return ret;
}

static long dvfel_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	struct dvfel_dev *d = gdev;
	void __user *p = (void __user *)arg;

	switch (cmd) {
	case DVFEL_IOC_REG_BUFS:
		return ioc_reg_bufs(d, f, p);
	case DVFEL_IOC_WAIT_JOB:
		return ioc_wait_job(d, f, p);
	case DVFEL_IOC_JOB_DONE:
		return ioc_job_done(d, f, p);
	case DVFEL_IOC_GET_POOL:
		return ioc_get_pool(d, f, p);
	case DVFEL_IOC_DISPLAY:
		return ioc_display(d, f, p);
	case DVFEL_IOC_UNREG_BUFS:
		mutex_lock(&d->client_lock);
		if (d->client == f) {
			client_unreg(d);
			d->client = NULL;
		}
		mutex_unlock(&d->client_lock);
		return 0;
	}
	return -ENOTTY;
}

static int dvfel_release(struct inode *inode, struct file *f)
{
	struct dvfel_dev *d = gdev;

	mutex_lock(&tvled_display_lock);
	if (d->display_owner == f && tvled_display_end(d)) {
		pr_warn(DRV_NAME ": display close restore failed\n");
		d->display_owner = NULL; /* no pointer to a released file */
	}
	mutex_unlock(&tvled_display_lock);
	mutex_lock(&d->client_lock);
	if (d->client == f) {
		client_unreg(d);
		d->client = NULL;
		pr_info(DRV_NAME ": compositor disconnected\n");
	}
	mutex_unlock(&d->client_lock);
	return 0;
}

static const struct file_operations dvfel_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = dvfel_ioctl,
	.compat_ioctl = dvfel_ioctl,
	.release = dvfel_release,
};

/* ------------------------------------------------------------------ */
/* debugfs                                                            */
/* ------------------------------------------------------------------ */

static int stats_show(struct seq_file *m, void *v)
{
	struct dvfel_dev *d = m->private;
	unsigned long flags;
	u64 delivered, unique, repeated, pts, pool_frames, token;
	s64 na = d->us_cnt ? d->us_cnt : 1;
	s64 nb = d->us_b_cnt ? d->us_b_cnt : 1;
	s64 ng = d->us_gpu_cnt ? d->us_gpu_cnt : 1;
	s64 nt = d->us_tvled_cnt ? d->us_tvled_cnt : 1;

	spin_lock_irqsave(&d->lock, flags);
	delivered = d->tvled_delivered;
	unique = d->tvled_unique_delivered;
	repeated = d->tvled_same_pts;
	pts = d->tvled_last_pts;
	pool_frames = d->tvled_pool_frames;
	token = d->pool_token;
	spin_unlock_irqrestore(&d->lock, flags);

	seq_printf(m, "mode %d, slots %d, provider %s, buffers %s (%ux%u), compositor %s\n",
		   mode, d->nslots, d->prov_reg ? "registered" : "idle",
		   d->bufs_ok ? "allocated" : "free", d->buf_w, d->buf_h,
		   d->client_ready ? "connected" : "none");
	seq_printf(m, "frames in %llu, processed %llu, passthrough %llu, vicp errors %llu, slot waits %llu, held breaks %llu\n",
		   d->frames_in, d->frames_proc, d->frames_pass, d->vicp_err,
		   d->slot_waits, d->held_breaks);
	seq_printf(m, "gpu jobs %llu, composed %llu, fallback %llu, timeouts %llu\n",
		   d->gpu_jobs, d->gpu_composed, d->gpu_fallback, d->gpu_timeouts);
	seq_printf(m, "tvled signal owned %d, vsif callback %d, native saved %d, native active saved %d\n",
		   d->signal_owned, d->vsif_on, d->signal_saved_enable,
		   d->signal_saved_native_on);
	seq_printf(m, "tvled active native claim rejects %llu\n", d->signal_active_rejects);
	seq_printf(m, "tvled carrier delivered this generation %d (driver milestone, not TV ack)\n",
		   READ_ONCE(d->tvled_started));
	seq_printf(m, "display_session %llu, epoch %llu, ready %d\n",
		   READ_ONCE(d->display_session), READ_ONCE(d->display_epoch),
		   READ_ONCE(d->display_ready));
	seq_printf(m, "tvled get delivered %llu, unique_pts %llu, repeated_pts %llu, last_pts_us %llu\n",
		   delivered, unique, repeated, pts);
	seq_printf(m, "tvled pool requested %d, ready %d, slots %d, token %llu, published %llu\n",
		   tvled_pool, READ_ONCE(d->pool_ready), d->nslots, token, pool_frames);
	seq_printf(m, "upstream pointer users BL %u, EL %u\n",
		   READ_ONCE(d->bl_users), READ_ONCE(d->el_users));
	seq_printf(m, "tvled requested %d, published %llu, dropped %llu\n",
		   tvled, d->tvled_frames, d->tvled_dropped);
	seq_printf(m, "tvled GE2D requested %d, copied %llu, CPU fallback %llu, byte verified %d, disabled %d\n",
		   tvled_ge2d, d->tvled_ge2d_frames, d->tvled_ge2d_fallback,
		   d->ge2d_verified, d->ge2d_failed);
	seq_printf(m, "tvled GE2D golden_checks %llu, context copies %u\n",
		   d->tvled_ge2d_golden_checks, d->ge2d_copies);
	seq_printf(m, "linear12 requested %d, published %llu, fallback %llu\n",
		   output12, d->linear12_frames, d->linear12_fallback);
	seq_printf(m, "VICP A (decompress)  avg %lld us max %lld us\n",
		   d->us_a_sum / na, d->us_a_max);
	seq_printf(m, "VICP B (compress)    avg %lld us max %lld us\n",
		   d->us_b_sum / nb, d->us_b_max);
	seq_printf(m, "GPU job (post->done) avg %lld us max %lld us\n",
		   d->us_gpu_sum / ng, d->us_gpu_max);
	seq_printf(m, "TVled CPU publish   avg %lld us max %lld us, samples %llu\n",
		   d->us_tvled_sum / nt, d->us_tvled_max, d->us_tvled_cnt);
	seq_printf(m, "EL (hardware) provider %s, client %ux%u: in %llu, used %llu, dropped %llu, missing %llu, held %d\n",
		   d->el_prov ? "registered" : "idle", d->client_el_w, d->client_el_h,
		   d->el_in, d->el_used, d->el_dropped, d->el_missing, d->el_nhold);
	seq_printf(m, "VICP A EL            avg %lld us max %lld us, EL wait max %lld us\n",
		   d->us_el_sum / (d->us_el_cnt ? d->us_el_cnt : 1), d->us_el_max,
		   d->us_el_wait_max);
	seq_printf(m, "queued %d, pending %d\n", d->outq_cnt, d->pend_cnt);
	{
		static const char * const recv[] = { DRV_NAME, EL_RECV_NAME };
		int i;

		for (i = 0; i < 2; i++) {
			struct vframe_provider_s *p = vf_get_provider(recv[i]);
			struct vframe_states st;

			if (p && !vf_get_states(p, &st))
				seq_printf(m, "%s provider %s: pool %d, avail %d, free %d, recycle %d\n",
					   recv[i], p->name, st.vf_pool_size, st.buf_avail_num,
					   st.buf_free_num, st.buf_recycle_num);
		}
	}
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(stats);

static ssize_t stats_reset_write(struct file *f, const char __user *buf,
				 size_t len, loff_t *ppos)
{
	struct dvfel_dev *d = f->private_data;
	unsigned long flags;

	d->frames_in = d->frames_proc = d->frames_pass = d->vicp_err = 0;
	d->slot_waits = d->held_breaks = 0;
	d->gpu_jobs = d->gpu_composed = d->gpu_fallback = d->gpu_timeouts = 0;
	d->linear12_frames = d->linear12_fallback = 0;
	d->tvled_frames = d->tvled_dropped = 0;
	spin_lock_irqsave(&d->lock, flags);
	d->tvled_pool_frames = 0;
	d->tvled_delivered = d->tvled_unique_delivered = d->tvled_same_pts = 0;
	d->tvled_last_pts = 0;
	d->tvled_have_pts = false;
	spin_unlock_irqrestore(&d->lock, flags);
	d->tvled_ge2d_frames = d->tvled_ge2d_fallback = 0;
	d->tvled_ge2d_golden_checks = 0;
	d->signal_active_rejects = 0;
	d->us_a_sum = d->us_b_sum = d->us_a_max = d->us_b_max = 0;
	d->us_gpu_sum = d->us_gpu_max = 0;
	d->us_tvled_sum = d->us_tvled_max = d->us_tvled_cnt = 0;
	d->el_in = d->el_used = d->el_dropped = d->el_missing = 0;
	d->us_el_sum = d->us_el_max = d->us_el_wait_max = d->us_el_cnt = 0;
	d->us_cnt = d->us_b_cnt = d->us_gpu_cnt = 0;
	return len;
}

static const struct file_operations stats_reset_fops = {
	.open = simple_open,
	.write = stats_reset_write,
};

/* write anything: capture the next processed frame */
static ssize_t capture_write(struct file *f, const char __user *buf,
			     size_t len, loff_t *ppos)
{
	struct dvfel_dev *d = f->private_data;

	atomic_set(&d->capture_req, 1);
	return len;
}

/*
 * read: 64-byte text header "DVFEL444P10 w h stride pts_us nframes",
 * then nframes linear 10-bit 4:4:4 frames
 */
static ssize_t capture_read(struct file *f, char __user *buf,
			    size_t len, loff_t *ppos)
{
	struct dvfel_dev *d = f->private_data;
	size_t frame = (size_t)LIN_STRIDE(d->cap_w) * d->cap_h;
	char hdr[64];
	loff_t pos;
	ssize_t r;
	int hl;

	if (!d->cap_frames)
		return 0;
	if (*ppos < 64) {
		hl = scnprintf(hdr, sizeof(hdr), "DVFEL444P10 %u %u %u %llu %d",
			       d->cap_w, d->cap_h, LIN_STRIDE(d->cap_w),
			       d->cap_pts_us, d->cap_frames);
		memset(hdr + hl, ' ', sizeof(hdr) - hl);
		hdr[63] = '\n';
		return simple_read_from_buffer(buf, len, ppos, hdr, 64);
	}
	pos = *ppos - 64;
	r = simple_read_from_buffer(buf, len, &pos, d->cap_buf, frame * d->cap_frames);
	if (r > 0)
		*ppos += r;
	return r;
}

static const struct file_operations capture_fops = {
	.open = simple_open,
	.read = capture_read,
	.write = capture_write,
	.llseek = default_llseek,
};

/* ------------------------------------------------------------------ */

static int __init dvfel_init(void)
{
	struct dvfel_dev *d;
	int ret;

	if (tvled_pool && clamp(slots, 2, MAX_SLOTS) > DVFEL_POOL_MAX_SLOTS)
		return -EINVAL;
	d = kzalloc(sizeof(*d), GFP_KERNEL);
	if (!d)
		return -ENOMEM;
	d->nslots = clamp(slots, 2, MAX_SLOTS);
	spin_lock_init(&d->lock);
	spin_lock_init(&d->el_lock);
	mutex_init(&d->buf_lock);
	mutex_init(&d->client_lock);
	init_waitqueue_head(&d->wq_a);
	init_waitqueue_head(&d->wq_b);
	init_waitqueue_head(&d->wq_client);
	init_waitqueue_head(&d->wq_bl_idle);
	init_waitqueue_head(&d->wq_el_idle);
	INIT_DELAYED_WORK(&d->free_work, dvfel_free_work);
	atomic_set(&d->capture_req, 0);
	gdev = d;

	d->misc.minor = MISC_DYNAMIC_MINOR;
	d->misc.name = DRV_NAME;
	d->misc.fops = &dvfel_fops;
	d->misc.mode = 0600;
	ret = misc_register(&d->misc);
	if (ret)
		goto err_free;
	dma_coerce_mask_and_coherent(d->misc.this_device, DMA_BIT_MASK(32));

	vf_receiver_init(&d->recv, DRV_NAME, &dvfel_recv_ops, d);
	vf_provider_init(&d->prov, DRV_NAME, &dvfel_vf_ops, d);

	d->thread_a = kthread_run(dvfel_thread_a, d, DRV_NAME "_a");
	if (IS_ERR(d->thread_a)) {
		ret = PTR_ERR(d->thread_a);
		goto err_misc;
	}
	d->thread_b = kthread_run(dvfel_thread_b, d, DRV_NAME "_b");
	if (IS_ERR(d->thread_b)) {
		ret = PTR_ERR(d->thread_b);
		kthread_stop(d->thread_a);
		goto err_misc;
	}
	vf_reg_receiver(&d->recv);
	vf_receiver_init(&d->el_recv, EL_RECV_NAME, &dvfel_el_recv_ops, d);
	vf_reg_receiver(&d->el_recv);

	d->dbg = debugfs_create_dir(DRV_NAME, NULL);
	debugfs_create_file("stats", 0444, d->dbg, d, &stats_fops);
	debugfs_create_file("stats_reset", 0200, d->dbg, d, &stats_reset_fops);
	debugfs_create_file("capture", 0600, d->dbg, d, &capture_fops);

	if (READ_ONCE(tvled_signal))
		tvled_signal_acquire(d, 0, true);
	pr_info(DRV_NAME ": loaded, mode %d, %d slots\n", mode, d->nslots);
	return 0;

err_misc:
	misc_deregister(&d->misc);
err_free:
	kfree(d);
	gdev = NULL;
	return ret;
}

static void __exit dvfel_exit(void)
{
	struct dvfel_dev *d = gdev;
	int ret;

	WRITE_ONCE(d->stopping, true);
	mutex_lock(&tvled_display_lock);
	if ((d->display_owner || d->display_native_saved) && tvled_display_end(d))
		pr_warn(DRV_NAME ": display exit restore failed\n");
	mutex_unlock(&tvled_display_lock);
	bl_quiesce(d, false);
	el_quiesce(d, false);
	tvled_signal_release(d);
	ret = tvled_attr_update(false);
	if (ret)
		pr_warn(DRV_NAME ": RGB8 preference exit restore failed (%d)\n", ret);
	debugfs_remove_recursive(d->dbg);
	vf_unreg_receiver(&d->recv);
	vf_unreg_receiver(&d->el_recv);
	/* prov_reg is now a closed input gate, not the output provider's
	 * registration state. VFM safely handles an already absent provider.
	 */
	vf_unreg_provider(&d->prov);
	misc_deregister(&d->misc);
	mutex_lock(&d->client_lock);
	if (d->client_nbufs)
		client_unreg(d);
	mutex_unlock(&d->client_lock);
	kthread_stop(d->thread_b);
	kthread_stop(d->thread_a);
	cancel_delayed_work_sync(&d->free_work);
	mutex_lock(&d->buf_lock);
	dvfel_free_bufs(d);
	mutex_unlock(&d->buf_lock);
	/* body_release_cb() must not outlive the module */
	if (atomic_read(&bodies_pending)) {
		int i;

		try_free_keep_video(1);
		for (i = 0; i < 50 && atomic_read(&bodies_pending); i++)
			msleep(20);
		if (atomic_read(&bodies_pending))
			pr_warn(DRV_NAME ": the video keeper still holds a frame\n");
	}
	vfree(d->cap_buf);
	gdev = NULL;
	kfree(d);
}

module_init(dvfel_init);
module_exit(dvfel_exit);

MODULE_DESCRIPTION("Dolby Vision FEL composition vfm node for Amlogic S5");
MODULE_LICENSE("GPL");
MODULE_IMPORT_NS(DMA_BUF);
