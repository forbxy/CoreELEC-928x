/* SPDX-License-Identifier: GPL-2.0-or-later WITH Linux-syscall-note */
/* Copyright (C) 2026 djnice <djnice@freemail.hu> */
/*
 * dvfel userspace interface (/dev/dvfel)
 *
 * A compositor process (Kodi) registers pairs of linear frame buffers
 * allocated from /dev/dma_heap/heap-codecmm. For every decoded frame dvfel
 * decompresses the base layer into in[buf] and posts a job; the compositor
 * writes the composed frame into out[buf] and reports completion. dvfel then
 * compresses out[buf] for display. Jobs are posted in display order.
 *
 * Buffer layout (both directions): width x height pixels, one little-endian
 * 32-bit word per pixel, stride = width * 4 bytes:
 *   bits  0- 9  Cr (10-bit)
 *   bits 10-19  Cb (10-bit)
 *   bits 20-29  Y  (10-bit)
 *   bits 30-31  0
 * In in[buf], chroma at even x and even y is the original 4:2:0 sample; odd
 * positions are interpolated. dvfel converts out[buf] back to 4:2:0 by
 * keeping the chroma at even x/y.
 *
 * For DVFEL_JOB_LINEAR12, out[buf] instead holds one little-endian 32-bit
 * word per pixel (the same width * 4 stride): bits 0..11 are Cb at even x
 * or Cr at odd x, bits 12..23 are Y, and bits 24..31 are ignored. All codes
 * are 12-bit, with 4:2:2 chroma. Complete with DVFEL_DONE_LINEAR12; dvfel
 * compacts the low three bytes into per-display-slot aligned memory.
 *
 * For DVFEL_JOB_TVLED, the same RGBA8 allocation carries the final RGB888
 * transport: byte 0 R=C>>4, byte 1 G=Y>>4, byte 2 B=(C&15)<<4|(Y&15).
 * These are logical RGB bytes; dvfel exchanges R/B when compacting to the
 * S5 display's B,G,R scanout memory. Metadata is already embedded by the
 * compositor. Complete with
 * DVFEL_DONE_TVLED. This format has priority over LINEAR12.
 *
 * Hardware decoded enhancement layer (Dolby Vision dual layer decoding, the
 * EL decoder feeding the "dvfelel" receiver): with el_width set, dvfel
 * decompresses the EL picture of each frame into el[buf] in the same layout
 * (el_width x el_height) and flags the job with DVFEL_JOB_EL.
 */
#ifndef _UAPI_DVFEL_H
#define _UAPI_DVFEL_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define DVFEL_MAX_BUFS		4
/* bytes needed per buffer (includes a write-MIF margin) */
#define DVFEL_BUF_SIZE(w, h)	((((w) * 4 * (h) + 256 * 1024) + 4095) & ~4095)

struct dvfel_reg_bufs {
	__u32 width;
	__u32 height;
	__u32 count;			/* 1..DVFEL_MAX_BUFS */
	__s32 in_fd[DVFEL_MAX_BUFS];	/* dma-buf fds */
	__s32 out_fd[DVFEL_MAX_BUFS];
	__u32 el_width;			/* 0: no hardware decoded EL */
	__u32 el_height;
	__s32 el_fd[DVFEL_MAX_BUFS];
};

#define DVFEL_JOB_NEW_STREAM	(1 << 0)	/* first job after start/seek */
#define DVFEL_JOB_EL		(1 << 1)	/* el[buf] holds the EL picture */
/* Bit 2 / status 2 were used by an incompatible experimental RGB tunnel. */
#define DVFEL_JOB_LINEAR12	(1 << 3)	/* compose 12-bit 422 words into out[buf] */
#define DVFEL_JOB_TVLED		(1 << 4)	/* compose RGB888 Dolby Vision carrier bytes */
/* This generation already handed a valid carrier to the display with the
 * VSIF callback active. A driver delivery milestone, not a television ack.
 * Absent on old kernels: clients may conservatively refresh startup DM.
 */
#define DVFEL_JOB_TVLED_STARTED	(1 << 5)
/* Opt-in direct output in driver-owned RGB24 display memory. The slot is
 * independent of the input/output pair index in job.buf. GET_POOL validates
 * this job's id and returns full token, dimensions and dma-buf FDs. Refresh
 * after NEW_STREAM, a cookie change or size change; pass token0 to obtain FDs.
 * A matching cached token returns metadata with every fd=-1 instead.
 * Write only the selected slot for this job, finish all GPU work before DONE5,
 * and finish/stop all GPU access before UNREG/close or replacing a cached pool.
 */
#define DVFEL_JOB_TVLED_POOL	(1U << 6)
#define DVFEL_JOB_POOL_SLOT_SHIFT	8
#define DVFEL_JOB_POOL_SLOT_MASK	(0xfU << DVFEL_JOB_POOL_SLOT_SHIFT)
#define DVFEL_JOB_POOL_COOKIE_SHIFT 16
#define DVFEL_JOB_POOL_COOKIE_MASK (0xffffU << DVFEL_JOB_POOL_COOKIE_SHIFT)
#define DVFEL_JOB_POOL_SLOT(flags) (((flags) >> DVFEL_JOB_POOL_SLOT_SHIFT) & 0xfU)
#define DVFEL_JOB_POOL_COOKIE(flags) (((flags) >> DVFEL_JOB_POOL_COOKIE_SHIFT) & 0xffffU)

struct dvfel_job {
	__u32 timeout_ms;		/* in: max wait */
	__u32 id;			/* out */
	__u32 buf;			/* out: buffer pair index */
	__u32 width;			/* out */
	__u32 height;			/* out */
	__u32 flags;			/* out: DVFEL_JOB_* */
	__u64 pts_us;			/* out: vframe pts_us64 (Kodi packet pts) */
	/*
	 * out: the access units (decode order, 16 bits, wrapping) of the BL
	 * frame and of its EL picture. They differ when the EL is coded in
	 * another picture order than the BL (paired by display order). These
	 * are diagnostic identities; RPU metadata in the supported single-track
	 * combo input follows the BL picture's presentation timestamp.
	 */
	__u32 bl_au;
	__u32 el_au;
};

#define DVFEL_DONE_COMPOSED	0	/* out[buf] holds the frame */
#define DVFEL_DONE_PASSTHROUGH	1	/* show in[buf] unchanged */
#define DVFEL_DONE_LINEAR12	3	/* low 24 bits: Y << 12 | Cb(even x)/Cr(odd x) */
#define DVFEL_DONE_TVLED		4	/* RGBA8: RGB bytes carry the 12-bit YCbCr tunnel */
#define DVFEL_DONE_TVLED_POOL	5	/* driver pool slot, requires job bit6 */

struct dvfel_job_done {
	__u32 id;
	__s32 status;			/* DVFEL_DONE_* */
};

#define DVFEL_POOL_MAX_SLOTS	8
struct dvfel_pool {
	__u32 job_id;			/* in: currently taken job.id */
	__u32 width;			/* out */
	__u32 height;			/* out */
	__u32 stride;			/* out: ALIGN(width*3,64), B,G,R bytes */
	__u32 count;			/* out: independent display slots, 1..8 */
	__u32 reserved;			/* in: zero */
	__u64 token;			/* in: cached token or0; out: full current token */
	__s32 fd[DVFEL_POOL_MAX_SLOTS]; /* out: CLOEXEC; -1 when cached/unused */
};

/* Separate display-owner fd; independent of the compositor's buffer fd.
 * BEGIN: session/epoch0, flags0, exact carrier raster; returns new ids.
 * PREPARE: closes TX readiness, completes native-inactive/RGB8 preparation;
 *          EBUSY may be retried with the unchanged epoch; success advances it.
 * READY: after the normal DRM mode/context commit; same raster and ids.
 * END: restore session-owned preferences; never clears VSIF during reset.
 * Every operation requires flags0. READY is idempotent for its current epoch.
 */
#define DVFEL_DISPLAY_BEGIN 1
#define DVFEL_DISPLAY_PREPARE 2
#define DVFEL_DISPLAY_READY 3
#define DVFEL_DISPLAY_END 4
struct dvfel_display {
	__u64 session;
	__u64 epoch;
	__u32 op;
	__u32 flags;
	__u32 width;
	__u32 height;
};

#define DVFEL_IOC_MAGIC		'F'
#define DVFEL_IOC_REG_BUFS	_IOW(DVFEL_IOC_MAGIC, 1, struct dvfel_reg_bufs)
#define DVFEL_IOC_WAIT_JOB	_IOWR(DVFEL_IOC_MAGIC, 2, struct dvfel_job)
#define DVFEL_IOC_JOB_DONE	_IOW(DVFEL_IOC_MAGIC, 3, struct dvfel_job_done)
#define DVFEL_IOC_UNREG_BUFS	_IO(DVFEL_IOC_MAGIC, 4)
#define DVFEL_IOC_GET_POOL	_IOWR(DVFEL_IOC_MAGIC, 5, struct dvfel_pool)
#define DVFEL_IOC_DISPLAY	_IOWR(DVFEL_IOC_MAGIC, 6, struct dvfel_display)

#endif
