#ifndef lint
static const char RCSid[] = "$Id: dctimestep.c,v 2.55 2025/06/07 05:09:46 greg Exp $";
#endif
/*
 * Compute time-step result using Daylight Coefficient method.
 * Optimized with AVX2, Mmap, and OpenMP Double Buffering Pipeline.
 */
#ifdef _WIN32
#define access _access
#define getcwd _getcwd
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#include <ctype.h>
#include <stdlib.h>
#include "platform.h"
#include "standard.h"
#include "cmatrix.h"
#include "resolu.h"

#if defined(__AVX2__)
#include <immintrin.h> 
#endif

#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "color.h"



typedef struct {
	void* data;
	size_t size;
#ifdef _WIN32
	HANDLE hFile;
	HANDLE hMap;
#else
	int fd;
#endif
} MappedFile;

MappedFile map_single_file(const char* path) {
	MappedFile mf = { NULL, 0 };
#ifdef _WIN32
	mf.hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (mf.hFile == INVALID_HANDLE_VALUE) return mf;

	LARGE_INTEGER liSize;
	GetFileSizeEx(mf.hFile, &liSize);
	mf.size = (size_t)liSize.QuadPart;

	mf.hMap = CreateFileMapping(mf.hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	if (mf.hMap == NULL) { CloseHandle(mf.hFile); return mf; }

	mf.data = MapViewOfFile(mf.hMap, FILE_MAP_READ, 0, 0, 0);
#else
	mf.fd = open(path, O_RDONLY);
	if (mf.fd == -1) return mf;

	struct stat st;
	fstat(mf.fd, &st);
	mf.size = st.st_size;

	mf.data = mmap(NULL, mf.size, PROT_READ, MAP_PRIVATE, mf.fd, 0);
	if (mf.data == MAP_FAILED) { mf.data = NULL; close(mf.fd); }
#endif
	return mf;
}

void unmap_file(MappedFile* mf) {
	if (!mf->data) return;
#ifdef _WIN32
	UnmapViewOfFile(mf->data);
	CloseHandle(mf->hMap);
	CloseHandle(mf->hFile);
#else
	munmap(mf->data, mf->size);
	close(mf->fd);
#endif
}

static int hasNumberFormat(const char* s) {
	if (s == NULL) return(0);
	while (*s) {
		while (*s != '%') if (!*s++) return(0);
		if (*++s == '%') { ++s; continue; }
		while (isdigit(*s)) ++s;
		if ((*s == 'd') | (*s == 'i') | (*s == 'o') | (*s == 'x') | (*s == 'X'))
			return(1);
	}
	return(0);
}

MappedFile* mapmultiHDRs(const char* fspec, int* outw, int* outh, int expected_images)
{
	char filepath[MAX_PATH];

	if (hasNumberFormat(fspec)) {
		snprintf(filepath, sizeof(filepath), fspec, 0);
	}
	else {
		strncpy(filepath, fspec, sizeof(filepath));
	}

	FILE* fp = fopen(filepath, "rb");
	if (!fp) {
		fprintf(stderr, "\n[Fatal Error] Cannot open HDR file '%s'.\n", filepath);
		return NULL;
	}

	int width, height;
	if (getheader(fp, NULL, NULL) < 0 || !fscnresolu(&width, &height, fp)) {
		fprintf(stderr, "[Fatal Error] Invalid Radiance HDR header in '%s'\n", filepath);
		fclose(fp);
		return NULL;
	}
	fclose(fp);

	*outw = width;
	*outh = height;

	MappedFile* sequence = malloc(sizeof(MappedFile) * expected_images);
	if (!sequence) return NULL;

	for (int i = 0; i < expected_images; i++) {
		char path[MAX_PATH];
		if (hasNumberFormat(fspec)) {
			snprintf(path, sizeof(path), fspec, i);
		}
		else {
			strncpy(path, fspec, sizeof(path));
		}

		sequence[i] = map_single_file(path);

		if (!sequence[i].data) {
			fprintf(stderr, "[Fatal Error] Failed to map file '%s'. Make sure exactly %d images exist.\n", path, expected_images);
			for (int j = 0; j < i; j++) unmap_file(&sequence[j]);
			free(sequence);
			return NULL;
		}
	}
	return sequence;
}

void unmapmultiHDRs(MappedFile* sequence, int num_images)
{
	for (int i = 0; i < num_images; i++) unmap_file(&sequence[i]);
	free(sequence);
}



float exponent_table[256];

void init_exponent_table() {
	for (int i = 0; i < 256; i++) {
		if (i == 0) exponent_table[i] = 0.0f;
		else exponent_table[i] = (float)ldexp(1.0, i - (128 + 8));
	}
}



int decode_hdr_line_planar(unsigned char** src, unsigned char* channels[4], int width) {
	unsigned char* p = *src;

	if (p[0] == 2 && p[1] == 2 && !(p[2] & 128)) {
		p += 4;

		for (int i = 0; i < 4; i++) {
			int j = 0;
			while (j < width) {
				unsigned char code = *p++;
				if (code > 128) {
					code &= 127;
					memset(&channels[i][j], *p++, code);
					j += code;
				}
				else {
					memcpy(&channels[i][j], p, code);
					p += code;
					j += code;
				}
			}
		}
	}
	else {

		for (int j = 0; j < width; j++) {
			channels[0][j] = *p++; // R
			channels[1][j] = *p++; // G
			channels[2][j] = *p++; // B
			channels[3][j] = *p++; // E
		}
	}

	*src = p;
	return 0;
}



void fill_matrix_chunk(MappedFile* sequence, int chunksize, int startid, int w, int h, CMATRIX* hdr_matrix) {

	hdr_matrix->nrows = w * h;
	hdr_matrix->ncols = chunksize;

	unsigned char** ptrs = (unsigned char**)malloc(chunksize * sizeof(unsigned char*));
	for (int c = 0; c < chunksize; c++) {
		unsigned char* ptr = sequence[startid + c].data;
		unsigned char* end = ptr + sequence[startid + c].size;

		while (ptr < end - 1) {
			if (ptr[0] == '\n' && ptr[1] == '\n') { ptr += 2; break; }
			if (ptr < end - 3 && ptr[0] == '\r' && ptr[1] == '\n' && ptr[2] == '\r' && ptr[3] == '\n') { ptr += 4; break; }
			ptr++;
		}
		while (ptr < end && *ptr != '\n') ptr++;
		if (ptr < end) ptr++;
		ptrs[c] = ptr;
	}

	unsigned char* planar_mem = (unsigned char*)malloc(4 * w);
	unsigned char* channels[4] = { planar_mem, planar_mem + w, planar_mem + 2 * w, planar_mem + 3 * w };
	COLORV* block_buffer = (COLORV*)malloc(w * chunksize * 3 * sizeof(COLORV));

	for (int y = 0; y < h; y++) {
		memset(block_buffer, 0, w * chunksize * 3 * sizeof(COLORV));

		for (int c = 0; c < chunksize; c++) {
			if (decode_hdr_line_planar(&ptrs[c], channels, w) == 0) {
				int x = 0;
#if defined(__AVX2__)
				for (; x <= w - 8; x += 8) {
					__m128i r_8 = _mm_loadl_epi64((__m128i const*) & channels[0][x]);
					__m128i g_8 = _mm_loadl_epi64((__m128i const*) & channels[1][x]);
					__m128i b_8 = _mm_loadl_epi64((__m128i const*) & channels[2][x]);
					__m128i e_8 = _mm_loadl_epi64((__m128i const*) & channels[3][x]);

					__m256i r_32 = _mm256_cvtepu8_epi32(r_8);
					__m256i g_32 = _mm256_cvtepu8_epi32(g_8);
					__m256i b_32 = _mm256_cvtepu8_epi32(b_8);
					__m256i e_32 = _mm256_cvtepu8_epi32(e_8);

					__m256 r_f = _mm256_cvtepi32_ps(r_32);
					__m256 g_f = _mm256_cvtepi32_ps(g_32);
					__m256 b_f = _mm256_cvtepi32_ps(b_32);

					__m256 exp_f = _mm256_i32gather_ps(exponent_table, e_32, 4);

					r_f = _mm256_mul_ps(r_f, exp_f);
					g_f = _mm256_mul_ps(g_f, exp_f);
					b_f = _mm256_mul_ps(b_f, exp_f);

					__m256i zero_mask = _mm256_cmpeq_epi32(e_32, _mm256_setzero_si256());
					__m256 zero_mask_ps = _mm256_castsi256_ps(zero_mask);

					r_f = _mm256_andnot_ps(zero_mask_ps, r_f);
					g_f = _mm256_andnot_ps(zero_mask_ps, g_f);
					b_f = _mm256_andnot_ps(zero_mask_ps, b_f);

					float R_arr[8], G_arr[8], B_arr[8];
					_mm256_storeu_ps(R_arr, r_f);
					_mm256_storeu_ps(G_arr, g_f);
					_mm256_storeu_ps(B_arr, b_f);

					for (int i = 0; i < 8; i++) {
						COLORV* p = &block_buffer[(x + i) * chunksize * 3 + c * 3];
						p[0] = R_arr[i];
						p[1] = G_arr[i];
						p[2] = B_arr[i];
					}
				}
#endif
				for (; x < w; x++) {
					unsigned char e = channels[3][x];
					if (e > 0) {
						float f = exponent_table[e];
						COLORV* p = &block_buffer[x * chunksize * 3 + c * 3];
						p[0] = (COLORV)(channels[0][x] * f);
						p[1] = (COLORV)(channels[1][x] * f);
						p[2] = (COLORV)(channels[2][x] * f);
					}
				}
			}
		}
		memcpy(&hdr_matrix->cmem[y * w * chunksize * 3], block_buffer, w * chunksize * 3 * sizeof(COLORV));
	}

	free(block_buffer);
	free(planar_mem);
	free(ptrs);
}



CMATRIX* sum_image_chunk(MappedFile* map, int start_idx, const CMATRIX* B, int total_images, int width, int height) {
	int total_pixels = width * height;
	int time_steps = B->ncols;

	if (B->nrows != total_images) {
		fprintf(stderr, "Fatal: Dimension mismatch! Sky Matrix has %d rows, but %d DC images found.\n", B->nrows, total_images);
		return NULL;
	}

	CMATRIX* final_result = cm_alloc_u(total_pixels, time_steps);
	memset(final_result->cmem, 0, (size_t)total_pixels * time_steps * 3 * sizeof(COLORV));

	int num_cores = omp_get_num_procs();
	unsigned long long MAX_MEMORY = 4ULL * 1024 * 1024 * 1024/num_cores;
	size_t frame_size = (size_t)total_pixels * 3 * sizeof(float);
	int MAX_CHUNK_SIZE = (int)((MAX_MEMORY / 2) / frame_size);
	if (MAX_CHUNK_SIZE < 1) MAX_CHUNK_SIZE = 1;

	fprintf(stderr, "Pipeline Max Chunk Size is %d\n", MAX_CHUNK_SIZE);


	CMATRIX* buffer[2];
	buffer[0] = cm_alloc_u(total_pixels, MAX_CHUNK_SIZE);
	buffer[1] = cm_alloc_u(total_pixels, MAX_CHUNK_SIZE);
	int buf_idx = 0;


	int first_chunk_size = (total_images > MAX_CHUNK_SIZE) ? MAX_CHUNK_SIZE : total_images;
	fprintf(stderr, "Pipeline Pre-fetching first block...\n");
	fill_matrix_chunk(map, first_chunk_size, 0, width, height, buffer[buf_idx]);





	for (int img_offset = 0; img_offset < total_images; img_offset += MAX_CHUNK_SIZE) {
		int current_chunk_size = total_images - img_offset;
		if (current_chunk_size > MAX_CHUNK_SIZE) current_chunk_size = MAX_CHUNK_SIZE;

		int comp_buf = buf_idx;
		buf_idx = 1 - buf_idx;


		int next_offset = img_offset + current_chunk_size;
		int next_chunk_size = total_images - next_offset;
		if (next_chunk_size > MAX_CHUNK_SIZE) next_chunk_size = MAX_CHUNK_SIZE;

		fprintf(stderr, "Pipeline Execute: Computing %d-%d | Pre-fetching next block\n",
			img_offset, img_offset + current_chunk_size - 1);


		#pragma omp parallel sections num_threads(2)
		{
			#pragma omp section
			{

				cm_multiply_accumulate_avx2(final_result, buffer[comp_buf], B, img_offset);
			}
			#pragma omp section
			{

				if (next_offset < total_images) {
					fill_matrix_chunk(map, next_chunk_size, next_offset, width, height, buffer[buf_idx]);
				}
			}
		}
	}

	cm_free_u(buffer[0]);
	cm_free_u(buffer[1]);

	return final_result;
}

CMATRIX* sum_image_smart(MappedFile* map, int start_idx, const CMATRIX* cv_batch, int width, int height, int total_images) {
	int R = width * height;
	int K = total_images;

	int C = cv_batch->ncols;
	int* is_zero_map = (int*)malloc(C * sizeof(int));
	int* nonzero_map = (int*)malloc(C * sizeof(int));
	int num_nonzero = 0;

	for (int c = 0; c < C; c++) {
		int zero_flag = 1;
		for (int k = 0; k < K; k++) {
			float* val = (float*)cm_lval(cv_batch, k, c);
			if (val[0] != 0.0f || val[1] != 0.0f || val[2] != 0.0f) {
				zero_flag = 0;
				break;
			}
		}
		is_zero_map[c] = zero_flag;
		if (!zero_flag) {
			nonzero_map[num_nonzero++] = c;
		}
	}

	if (num_nonzero == 0) {
		CMATRIX* final_res = cm_alloc_u(R, C);
		memset(final_res->cmem, 0, (size_t)R * C * 3 * sizeof(COLORV));
		free(is_zero_map); free(nonzero_map);
		return final_res;
	}

	if (num_nonzero == C) {
		free(is_zero_map); free(nonzero_map);
		return sum_image_chunk(map, start_idx, cv_batch, total_images, width, height);
	}

	CMATRIX* compact_cv = cm_alloc_u(K, num_nonzero);
	for (int nz = 0; nz < num_nonzero; nz++) {
		int original_col = nonzero_map[nz];
		for (int k = 0; k < K; k++) {
			float* src = (float*)cm_lval(cv_batch, k, original_col);
			float* dst = (float*)cm_lval(compact_cv, k, nz);
			dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
		}
	}

	CMATRIX* compact_res = sum_image_chunk(map, start_idx, compact_cv, total_images, width, height);
	cm_free_u(compact_cv);
	if (!compact_res) {
		free(is_zero_map); free(nonzero_map);
		return NULL;
	}

	CMATRIX* final_res = cm_alloc_u(R, C);
	if (final_res) {
		memset(final_res->cmem, 0, (size_t)R * C * 3 * sizeof(COLORV));

		for (int r = 0; r < R; r++) {
			for (int nz = 0; nz < num_nonzero; nz++) {
				int original_col = nonzero_map[nz];
				float* src = (float*)cm_lval(compact_res, r, nz);
				float* dst = (float*)cm_lval(final_res, r, original_col);

				dst[0] = src[0];
				dst[1] = src[1];
				dst[2] = src[2];
			}
		}
	}

	cm_free_u(compact_res);
	free(is_zero_map);
	free(nonzero_map);

	return final_res;
}



#ifndef _WIN32
void set_max_files(int num_files) {
	struct rlimit rl;
	if (getrlimit(RLIMIT_NOFILE, &rl) == -1) {
		perror("getrlimit");
		return;
	}
	rl.rlim_cur = num_files;
	if (num_files > rl.rlim_max) {
		rl.rlim_max = num_files;
	}
	if (setrlimit(RLIMIT_NOFILE, &rl) == -1) {
		perror("setrlimit");
	}
}
#endif

int* stat_numn_onzerocol(CMATRIX* cv, int* size) {
	int K = cv->nrows;
	int C = cv->ncols;

	int* pixel_count_map = (int*)calloc(C, sizeof(int));
	int num_nonzero = 0;

	for (int c = 0; c < C; c++) {
		int col_pixel_count = 0;
		for (int k = 0; k < K; k++) {
			float* val = (float*)cm_lval(cv, k, c);
			if (val[0] != 0.0f || val[1] != 0.0f || val[2] != 0.0f) {
				col_pixel_count++;
			}
		}

		if (col_pixel_count > 0) {
			pixel_count_map[num_nonzero] = col_pixel_count;
			num_nonzero++;
		}
	}

	*size = num_nonzero;
	return pixel_count_map;
}

static int sum_images_matrix(MappedFile* map, int total_images, int w, int h, const CMATRIX* cv, FILE* fout) {
	CMATRIX* pmat = sum_image_smart(map, 0, cv, w, h, total_images);
	if (!pmat) {
		printf("pmat is null\n");
		return 0;
	}

	int save_r = pmat->nrows;
	int save_c = pmat->ncols;
	pmat->nrows = h;
	pmat->ncols = w;

	fputformat((char*)cm_fmt_id[DTrgbe], fout);
	fputc('\n', fout);
	fflush(fout);

	int i = cm_write(pmat, DTrgbe, fout);

	pmat->nrows = save_r;
	pmat->ncols = save_c;

	cm_free_u(pmat);
	return i;
}

static int sum_images_matrix_batch(MappedFile* map, const CMATRIX* cv_batch, int start_index, int total_images, int w, int h, FILE* ofspec_base, const char* ofspec_pattern) {
	CMATRIX* pmat_batch = sum_image_smart(map, start_index, cv_batch, w, h, total_images);
	if (!pmat_batch) {
		fprintf(stderr, "Matrix multiplication failed for batch starting at %d\n", start_index);
		return 0;
	}

	int batch_size = pmat_batch->ncols;
	for (int b = 0; b < batch_size; b++) {
		char fnbuf[1024];
		FILE* ofp = ofspec_base;
		int current_frame_id = start_index + b;
		int needs_fclose = 0;

		if (ofspec_pattern != NULL && hasNumberFormat(ofspec_pattern)) {
			sprintf(fnbuf, ofspec_pattern, current_frame_id);
			if ((ofp = fopen(fnbuf, "wb")) == NULL) {
				fprintf(stderr, "%s: cannot open '%s' for output\n", progname, fnbuf);
				return(1);
			};
			needs_fclose = 1;
			newheader("RADIANCE", ofp);
			fputnow(ofp);
			fputformat((char*)cm_fmt_id[DTrgbe], ofp);
			fprintf(ofp, "FRAME=%d\n", current_frame_id);
			fputc('\n', ofp);
			fflush(ofp);
		}

		if (ofp != NULL) {
			CMATRIX* one_frame = cm_column_u(pmat_batch, b);
			int save_r = one_frame->nrows;
			int save_c = one_frame->ncols;
			one_frame->nrows = h;
			one_frame->ncols = w;

#pragma omp critical(hdr_encode_lock)
			{
				if (cm_write(one_frame, DTrgbe, ofp) < 0) {
					fprintf(stderr, "Error writing frame %d\n", current_frame_id);
				}
			}

			one_frame->nrows = save_r;
			one_frame->ncols = save_c;
			cm_free_u(one_frame);
		}

		if (needs_fclose) {
			fclose(ofp);
		}
	}
	cm_free_u(pmat_batch);
	return 1;
}

static int sum_images(const char* fspec, const CMATRIX* cv, FILE* fout) {
	static int	runcnt = 0;
	int	myDT = DTfromHeader;
	COLR* scanline = NULL;
	CMATRIX* pmat = NULL;
	int	myXR = 0, myYR = 0;
	int	i, y;

	if (cv->ncols != 1)
		error(INTERNAL, "expected vector in sum_images()");
	for (i = cv->nrows; i-- > 0; ) {
		const int	r = runcnt & 1 ? i : cv->nrows - 1 - i;
		const COLORV* scv = cv_lval(cv, r);
		int		flat_file = 0;
		char		fname[1024];
		FILE* fp;
		long		data_start;
		int		dt, xr, yr;
		COLORV* psp;
		char* err;

		if ((scv[RED] == 0) & (scv[GRN] == 0) & (scv[BLU] == 0) &&
			(myDT != DTfromHeader) | (i > 0))
			continue;

		sprintf(fname, fspec, r);
		if ((fp = fopen(fname, "rb")) == NULL) {
			sprintf(errmsg, "cannot open picture '%s'", fname);
			error(SYSTEM, errmsg);
		}
#ifdef getc_unlocked
		flockfile(fp);
#endif
		dt = DTfromHeader;
		if ((err = cm_getheader(&dt, NULL, NULL, NULL, NULL, fp)) != NULL)
			error(USER, err);
		if ((dt != DTrgbe) & (dt != DTxyze) || !fscnresolu(&xr, &yr, fp)) {
			sprintf(errmsg, "file '%s' not a picture", fname);
			error(USER, errmsg);
		}
		if (myDT == DTfromHeader) {
			myDT = dt;
			myXR = xr; myYR = yr;
			scanline = (COLR*)malloc(sizeof(COLR) * myXR);
			if (scanline == NULL) error(SYSTEM, "out of memory in sum_images()");
			pmat = cm_alloc(myYR, myXR);
			memset(pmat->cmem, 0, sizeof(COLOR) * myXR * myYR);

			fputformat(cm_fmt_id[myDT], fout);
			fputc('\n', fout);
			fflush(fout);
		}
		else if ((dt != myDT) | (xr != myXR) | (yr != myYR)) {
			sprintf(errmsg, "picture '%s' format/size mismatch", fname);
			error(USER, errmsg);
		}

		if ((data_start = ftell(fp)) > 0 && fseek(fp, 0L, SEEK_END) == 0) {
			flat_file = (ftell(fp) >= data_start + sizeof(COLR) * xr * yr);
			if (fseek(fp, data_start, SEEK_SET) < 0) {
				sprintf(errmsg, "cannot seek on picture '%s'", fname);
				error(SYSTEM, errmsg);
			}
		}
		psp = pmat->cmem;
		for (y = 0; y < yr; y++) {
			COLOR	col;
			int	x;
			if (flat_file ? getbinary(scanline, sizeof(COLR), xr, fp) != xr :
				freadcolrs(scanline, xr, fp) < 0) {
				sprintf(errmsg, "error reading picture '%s'", fname);
				error(SYSTEM, errmsg);
			}
			for (x = 0; x < xr; x++, psp += 3) {
				if (!scanline[x][EXP]) continue;
				colr_color(col, scanline[x]);
				multcolor(col, scv);
				addcolor(psp, col);
			}
		}
		fclose(fp);
	}
	free(scanline);
	i = cm_write(pmat, myDT, fout);
	cm_free(pmat);
	++runcnt;
	return(i);
}

static int alt_dim(CMATRIX* cm, int nr, int nc) {
	if ((nr <= 0) & (nc <= 0)) return(0);
	if ((nr == cm->nrows) & (nc == cm->ncols)) return(0);
	if (nr > 0) {
		if (nc <= 0) nc = cm->nrows * cm->ncols / nr;
		if (nr * nc != cm->nrows * cm->ncols) {
			fprintf(stderr, "Bad dimensions: %dx%d != %dx%d\n", nr, nc, cm->nrows, cm->ncols);
			return(-1);
		}
	}
	else {
		nr = cm->nrows * cm->ncols / nc;
		if (nc * nr != cm->nrows * cm->ncols) {
			fprintf(stderr, "Bad dimensions: %d does not divide %dx%d evenly\n", nc, cm->nrows, cm->ncols);
			return(-1);
		}
	}
	cm->nrows = nr;
	cm->ncols = nc;
	return(1);
}

int main(int argc, char* argv[])
{
#if _OPENMP >= 201511
	omp_set_max_active_levels(2);
#else
	omp_set_nested(1);
#endif

	int		skyfmt = DTfromHeader;
	int		outfmt = DTascii;
	int		headout = 1;
	int		nsteps = 0;
	char* ofspec = NULL;
	FILE* ofp = stdout;
	int		xres = 0, yres = 0;
	CMATRIX* cmtx;
	char		fnbuf[256];
	int		a, i;

	fixargv0(argv[0]);

	for (a = 1; a < argc && argv[a][0] == '-'; a++)
		switch (argv[a][1]) {
		case 'n':
			nsteps = atoi(argv[++a]);
			if (nsteps < 0) goto userr;
			skyfmt = nsteps ? DTascii : DTfromHeader;
			break;
		case 'h':
			headout = !headout;
			break;
		case 'i':
			switch (argv[a][2]) {
			case 'f': skyfmt = DTfloat; break;
			case 'd': skyfmt = DTdouble; break;
			case 'a': skyfmt = DTascii; break;
			default: goto userr;
			}
			break;
		case 'o':
			switch (argv[a][2]) {
			case '\0': ofspec = argv[++a]; ofspec = "results/2ph/%04d.hdr";break;
			case 'f': outfmt = DTfloat; break;
			case 'd': outfmt = DTdouble; break;
			case 'a': outfmt = DTascii; break;
			case 'c': outfmt = DTrgbe; break;
			default: goto userr;
			}
			break;
		case 'x': xres = atoi(argv[++a]); break;
		case 'y': yres = atoi(argv[++a]); break;
		default: goto userr;
		}
	if ((argc - a < 1) | (argc - a > 4)) goto userr;


	init_exponent_table();

	if (argc - a > 2) {
		CMATRIX* smtx, * Dmat, * Tmat, * imtx;
		const char* ccp;

		smtx = cm_load_u(argv[a + 3], 0, nsteps, skyfmt);
		nsteps = smtx->ncols;

		if (argv[a + 1][0] != '!' && (ccp = strrchr(argv[a + 1], '.')) > argv[a + 1] && !strcasecmp(ccp + 1, "XML"))
			Tmat = cm_loadBTDF(argv[a + 1]);
		else
			Tmat = cm_load_u(argv[a + 1], 0, 0, DTfromHeader);

		Dmat = cm_load_u(argv[a + 2], Tmat->ncols, smtx->nrows, DTfromHeader);
		imtx = cm_multiply_smart(Dmat, smtx);
		cm_free_u(Dmat); cm_free_u(smtx);
		cmtx = cm_multiply_smart(Tmat, imtx);
		cm_free(Tmat);
		cm_free_u(imtx);
	}
	else {
		cmtx = cm_load_u(argv[a + 1], 0, nsteps, skyfmt);
		nsteps = cmtx->ncols;
	}

	if ((ofspec != NULL) & (nsteps == 1) && hasNumberFormat(ofspec)) {
		sprintf(fnbuf, ofspec, 0);
		ofspec = fnbuf;
	}
	if (ofspec != NULL && !hasNumberFormat(ofspec)) {
		if ((ofp = fopen(ofspec, "w")) == NULL) {
			fprintf(stderr, "%s: cannot open '%s' for output\n", progname, ofspec);
			return(1);
		}
		ofspec = NULL;
	}
	argv[a] = "matrices/dcm3/%04d.hdr";
	if (hasNumberFormat(argv[a])) {
		if (outfmt != DTrgbe) {
			error(WARNING, "changing output type to -oc");
			outfmt = DTrgbe;
		}
		if (ofspec == NULL) {
			SET_FILE_BINARY(ofp);
			newheader("RADIANCE", ofp);
			printargs(argc, argv, ofp);
			fputnow(ofp);
		}

		if (nsteps > 1) {
			int batch_size = 64;
			int size = 0;
			int* nonzeronum_percols = stat_numn_onzerocol(cmtx, &size);
			int allone = 1;
			for (int k = 0; k < size; k++) {
				if (*(nonzeronum_percols + k) != 1) {
					allone = 0;
					break;
				}
			}

			if (allone == 0) {
				int cached_w = 0, cached_h = 0;

				int total_images = cmtx->nrows;

				MappedFile* map = mapmultiHDRs(argv[a], &cached_w, &cached_h, total_images);
				if (!map) return(1);

				#pragma omp parallel for num_threads(16) schedule(dynamic, 1) private(i)
				for (i = 0; i < nsteps; i += batch_size) {
					FILE* batch_out_fp = (ofspec == NULL) ? ofp : NULL;
					int current_batch = (i + batch_size > nsteps) ? (nsteps - i) : batch_size;
					CMATRIX* cvec_batch = cm_get_batch(cmtx, i, current_batch);
					if (!sum_images_matrix_batch(map, cvec_batch, i, total_images, cached_w, cached_h, batch_out_fp, ofspec)) {
						error(SYSTEM, "Batch matrix calculation failed");
					}
					cm_free_u(cvec_batch);
				}
				unmapmultiHDRs(map, total_images);

			}
			else {
				for (i = 0; i < nsteps; i++) {
					CMATRIX* cvec = cm_column(cmtx, i);
					if (ofspec != NULL) {
						sprintf(fnbuf, ofspec, i);
						if ((ofp = fopen(fnbuf, "wb")) == NULL) {
							fprintf(stderr, "%s: cannot open '%s' for output\n", progname, fnbuf);
							return(1);
						}
						newheader("RADIANCE", ofp);
						printargs(argc, argv, ofp);
						fputnow(ofp);
					}
					fprintf(ofp, "FRAME=%d\n", i);
					if (!sum_images(argv[a], cvec, ofp)) return(1);
					if (ofspec != NULL) {
						if (fclose(ofp) == EOF) {
							fprintf(stderr, "%s: error writing to '%s'\n", progname, fnbuf);
							return(1);
						}
						ofp = stdout;
					}
					cm_free(cvec);
				}
			}
		}
		else {
			int size = 0;
			int* nonzeronum_percols = stat_numn_onzerocol(cmtx, &size);
			int allone = 1;
			for (int k = 0; k < size; k++) {
				if (*(nonzeronum_percols + k) != 1) {
					allone = 0;
					break;
				}
			}

			if (allone == 0) {
				int cached_w = 0, cached_h = 0;
				int total_images = cmtx->nrows;

				MappedFile* map = mapmultiHDRs(argv[a], &cached_w, &cached_h, total_images);
				if (!map) return(1);

				if (!sum_images_matrix(map, total_images, cached_w, cached_h, cmtx, ofp)) return(1);
			}
			else {
				if (!sum_images(argv[a], cmtx, ofp)) return (1);
			}
		}
	}
	else {
		CMATRIX* Vmat = cm_load_u(argv[a], 0, cmtx->nrows, DTfromHeader);
		CMATRIX* rmtx = cm_multiply_smart(Vmat, cmtx);
		cm_free_u(Vmat);
		if (ofspec != NULL) {
			const char* wtype = (outfmt == DTascii) ? "w" : "wb";
			for (i = 0; i < nsteps; i++) {
				CMATRIX* rvec = cm_column_u(rmtx, i);
				if (alt_dim(rvec, yres, xres) < 0) return(1);
				sprintf(fnbuf, ofspec, i);
				if ((ofp = fopen(fnbuf, wtype)) == NULL) {
					fprintf(stderr, "%s: cannot open '%s' for output\n", progname, fnbuf);
					return(1);
				}
#ifdef getc_unlocked
				flockfile(ofp);
#endif
				if (headout) {
					newheader("RADIANCE", ofp);
					printargs(argc, argv, ofp);
					fputnow(ofp);
					fprintf(ofp, "FRAME=%d\n", i);
					if ((outfmt != DTrgbe) & (outfmt != DTxyze)) {
						fprintf(ofp, "NROWS=%d\n", rvec->nrows);
						fprintf(ofp, "NCOLS=%d\n", rvec->ncols);
						fputncomp(3, ofp);
					}
					if ((outfmt == DTfloat) | (outfmt == DTdouble)) fputendian(ofp);
					fputformat(cm_fmt_id[outfmt], ofp);
					fputc('\n', ofp);
				}
				cm_write(rvec, outfmt, ofp);
				if (fclose(ofp) == EOF) {
					fprintf(stderr, "%s: error writing to '%s'\n", progname, fnbuf);
					return(1);
				}
				ofp = stdout;
				cm_free(rvec);
			}
		}
		else {
#ifdef getc_unlocked
			flockfile(ofp);
#endif
			if (outfmt != DTascii) SET_FILE_BINARY(ofp);
			if (alt_dim(rmtx, yres, xres) < 0) return(1);
			if (headout) {
				newheader("RADIANCE", ofp);
				printargs(argc, argv, ofp);
				fputnow(ofp);
				if ((outfmt != DTrgbe) & (outfmt != DTxyze)) {
					fprintf(ofp, "NROWS=%d\n", rmtx->nrows);
					fprintf(ofp, "NCOLS=%d\n", rmtx->ncols);
					fputncomp(3, ofp);
				}
				if ((outfmt == DTfloat) | (outfmt == DTdouble)) fputendian(ofp);
				fputformat(cm_fmt_id[outfmt], ofp);
				fputc('\n', ofp);
			}
			cm_write(rmtx, outfmt, ofp);
		}
		cm_free_u(rmtx);
	}
	if (fflush(ofp) == EOF) {
		fprintf(stderr, "%s: write error on output\n", progname);
		return(1);
	}
	cm_free_u(cmtx);
	return(0);
userr:
	fprintf(stderr, "Usage: %s [-n nsteps][-o ospec][-x xr][-y yr][-i{f|d|h}][-o{f|d|c}] DCspec [skyf]\n", progname);
	fprintf(stderr, "   or: %s [-n nsteps][-o ospec][-x xr][-y yr][-i{f|d|h}][-o{f|d|c}] Vspec Tbsdf Dmat.dat [skyf]\n", progname);
	return(1);
}