// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#include <chiaki/feedbacksender.h>
#include <chiaki/time.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* Keep observation separate from the protocol clock. With diagnostics disabled
 * none of these QPC calls, sample allocations or file writes take place. */
#define FEEDBACK_DIAGNOSTIC_CAPACITY 4096
enum feedback_diagnostic_stage {
	FD_FOREGROUND_WAIT, FD_FOREGROUND_STATE, FD_FOREGROUND_HISTORY,
	FD_REPEAT_STATE, FD_REPEAT_HISTORY, FD_REPEAT_HOLD, FD_STAGE_COUNT
};
typedef struct feedback_diagnostic_sample {
	uint64_t start, duration;
} FeedbackDiagnosticSample;
typedef struct feedback_diagnostic_series {
	uint64_t total_count, total_ticks, max_ticks;
	FeedbackDiagnosticSample samples[FEEDBACK_DIAGNOSTIC_CAPACITY];
} FeedbackDiagnosticSeries;
typedef struct chiaki_feedback_diagnostics_t {
	uint64_t frequency, origin;
	bool repeat;
	FeedbackDiagnosticSeries stages[FD_STAGE_COUNT];
} ChiakiFeedbackDiagnostics;

static uint64_t feedback_diagnostic_clock(void)
{
#ifdef _WIN32
	LARGE_INTEGER now;
	return QueryPerformanceCounter(&now) ? (uint64_t)now.QuadPart : 0;
#else
	return 0;
#endif
}

static ChiakiFeedbackDiagnostics *feedback_diagnostics_create(void)
{
#ifdef _WIN32
	wchar_t enabled[4];
	if(GetEnvironmentVariableW(L"PS5BRIDGE_DIAGNOSTICS", enabled, 4) != 1 || enabled[0] != L'1')
		return NULL;
	ChiakiFeedbackDiagnostics *diagnostics = calloc(1, sizeof(*diagnostics));
	if(!diagnostics)
		return NULL;
	LARGE_INTEGER frequency;
	if(!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0)
	{
		free(diagnostics);
		return NULL;
	}
	diagnostics->frequency = (uint64_t)frequency.QuadPart;
	diagnostics->origin = feedback_diagnostic_clock();
	return diagnostics;
#else
	return NULL;
#endif
}

/* Caller owns state_mutex; the worker and input thread never mutate a series
 * concurrently. No diagnostics mutex or floating-point work on the send path. */
static void feedback_diagnostic_record(ChiakiFeedbackDiagnostics *diagnostics,
		enum feedback_diagnostic_stage stage, uint64_t start, uint64_t end)
{
	if(!diagnostics || !start || end < start)
		return;
	FeedbackDiagnosticSeries *series = &diagnostics->stages[stage];
	uint64_t duration = end - start;
	series->samples[series->total_count % FEEDBACK_DIAGNOSTIC_CAPACITY] =
		(FeedbackDiagnosticSample){start, duration};
	series->total_count++;
	series->total_ticks += duration;
	if(duration > series->max_ticks)
		series->max_ticks = duration;
}

static void feedback_diagnostics_write(ChiakiFeedbackDiagnostics *diagnostics)
{
	if(!diagnostics)
		return;
#ifdef _WIN32
	wchar_t directory[1024], path[1200];
	DWORD count = GetEnvironmentVariableW(L"PS5BRIDGE_DIAGNOSTICS_DIR", directory, 1024);
	if(count)
	{
		/* Do not resolve a relative override against an arbitrary launch folder. */
		if(count >= 1024 || !((count >= 3 && directory[1] == L':'
				&& (directory[2] == L'\\' || directory[2] == L'/'))
				|| (count >= 3 && directory[0] == L'\\' && directory[1] == L'\\')))
			return;
		CreateDirectoryW(directory, NULL);
	}
	else
	{
		wchar_t local[1024];
		count = GetEnvironmentVariableW(L"LOCALAPPDATA", local, 1024);
		if(!count || count >= 950)
			return;
		_snwprintf(directory, 1024, L"%ls\\PS5InputBridge", local);
		CreateDirectoryW(directory, NULL);
		_snwprintf(directory, 1024, L"%ls\\PS5InputBridge\\diagnostics", local);
		CreateDirectoryW(directory, NULL);
	}
	DWORD attributes = GetFileAttributesW(directory);
	if(attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY))
		return;
	FILETIME now;
	GetSystemTimeAsFileTime(&now);
	uint64_t timestamp = ((uint64_t)now.dwHighDateTime << 32) | now.dwLowDateTime;
	static volatile LONG serial;
	int length = _snwprintf(path, 1200, L"%ls\\feedback-%lu-%llu-%lu.csv", directory,
		(unsigned long)GetCurrentProcessId(), (unsigned long long)timestamp,
		(unsigned long)InterlockedIncrement(&serial));
	if(length < 0 || length >= 1200)
		return;
	FILE *file = _wfopen(path, L"w");
	if(!file)
		return;
	static const char *names[FD_STAGE_COUNT] = {
		"foreground_mutex_wait", "foreground_state_send", "foreground_history_send",
		"repeat_state_send", "repeat_history_send", "repeat_active_hold"
	};
	double scale = 1000000.0 / (double)diagnostics->frequency;
	fprintf(file, "kind,stage,sample_index,start_us,duration_us,total_count,total_duration_us,max_duration_us\n");
	for(unsigned stage = 0; stage < FD_STAGE_COUNT; stage++)
	{
		FeedbackDiagnosticSeries *series = &diagnostics->stages[stage];
		fprintf(file, "summary,%s,,,,%llu,%.3f,%.3f\n", names[stage],
			(unsigned long long)series->total_count, (double)series->total_ticks * scale,
			(double)series->max_ticks * scale);
		uint64_t first = series->total_count > FEEDBACK_DIAGNOSTIC_CAPACITY
			? series->total_count - FEEDBACK_DIAGNOSTIC_CAPACITY : 0;
		for(uint64_t i = first; i < series->total_count; i++)
		{
			FeedbackDiagnosticSample *sample = &series->samples[i % FEEDBACK_DIAGNOSTIC_CAPACITY];
			fprintf(file, "sample,%s,%llu,%.3f,%.3f,,,\n", names[stage],
				(unsigned long long)i, (double)(sample->start - diagnostics->origin) * scale,
				(double)sample->duration * scale);
		}
	}
	fclose(file);
#endif
}



#define FEEDBACK_STATE_TIMEOUT_MIN_MS 8 // minimum time to wait between sending 2 packets
#define FEEDBACK_STATE_TIMEOUT_MAX_MS 200 // maximum time to wait between sending 2 packets

#define FEEDBACK_HISTORY_BUFFER_SIZE 0x10
// Number of most recent events repeated in every history packet.
// Analog L2/R2 produce one event per value change, so with only 4 a button release
// (e.g. L1/R1) could drop out of the window within one or two packets.
#define FEEDBACK_HISTORY_RESEND_EVENT_COUNT 0x8

// History packets are sent once over UDP without acknowledgement. If the packet carrying
// the last change (typically a button release) is lost and nothing else changes afterwards,
// the console never sees it and the button stays held. So after the newest history packet,
// if nothing new is queued, the same packet is sent again at these offsets (ms after it was sent).
// This is safe: events carry absolute values (button 0/0xff, trigger 0..255, touch position)
// and only the newest packet is repeated, so state can never go backwards.
static const uint64_t feedback_history_tail_ms[] = { 20, 50, 100, 200 };
#define FEEDBACK_HISTORY_TAIL_COUNT (sizeof(feedback_history_tail_ms) / sizeof(feedback_history_tail_ms[0]))

static void *feedback_sender_thread_func(void *user);
static ChiakiErrorCode feedback_sender_send_state(ChiakiFeedbackSender *feedback_sender, const ChiakiControllerState *state);
static ChiakiErrorCode feedback_sender_send_history_packet(ChiakiFeedbackSender *feedback_sender, uint8_t event_count, const uint8_t *buf, size_t buf_size);
static ChiakiErrorCode feedback_sender_send_now_locked(ChiakiFeedbackSender *feedback_sender);
static void feedback_sender_flush_history_locked(ChiakiFeedbackSender *feedback_sender);
static void feedback_sender_record_history(ChiakiFeedbackSender *feedback_sender, const ChiakiControllerState *state_prev, const ChiakiControllerState *state_now);

CHIAKI_EXPORT ChiakiErrorCode chiaki_feedback_sender_init(ChiakiFeedbackSender *feedback_sender, ChiakiTakion *takion)
{
	feedback_sender->log = takion->log;
	feedback_sender->takion = takion;
	feedback_sender->diagnostics = NULL;

	chiaki_controller_state_set_idle(&feedback_sender->controller_state_prev);
	chiaki_controller_state_set_idle(&feedback_sender->controller_state_history_prev);
	chiaki_controller_state_set_idle(&feedback_sender->controller_state);

	feedback_sender->state_seq_num = 0;

	feedback_sender->history_seq_num = 0;
	feedback_sender->history_packet_begin = 0;
	feedback_sender->history_packet_len = 0;
	feedback_sender->should_stop = false;
	feedback_sender->controller_state_changed = false;
	feedback_sender->history_dirty = false;
	feedback_sender->history_last_size = 0;
	feedback_sender->history_last_event_count = 0;
	feedback_sender->history_last_sent_ms = 0;
	feedback_sender->history_tail_resends = FEEDBACK_HISTORY_TAIL_COUNT;
	ChiakiErrorCode err = chiaki_feedback_history_buffer_init(&feedback_sender->history_buf, FEEDBACK_HISTORY_BUFFER_SIZE);
	if(err != CHIAKI_ERR_SUCCESS)
		return err;

	err = chiaki_mutex_init(&feedback_sender->state_mutex, false);
	if(err != CHIAKI_ERR_SUCCESS)
		goto error_history_buffer;

	err = chiaki_cond_init(&feedback_sender->state_cond);
	if(err != CHIAKI_ERR_SUCCESS)
		goto error_mutex;

	feedback_sender->diagnostics = feedback_diagnostics_create();
	err = chiaki_thread_create(&feedback_sender->thread, feedback_sender_thread_func, feedback_sender);
	if(err != CHIAKI_ERR_SUCCESS)
		goto error_cond;

	chiaki_thread_set_name(&feedback_sender->thread, "Chiaki Feedback Sender");

	return CHIAKI_ERR_SUCCESS;
error_cond:
	free(feedback_sender->diagnostics);
	feedback_sender->diagnostics = NULL;
	chiaki_cond_fini(&feedback_sender->state_cond);
error_mutex:
	chiaki_mutex_fini(&feedback_sender->state_mutex);
error_history_buffer:
	chiaki_feedback_history_buffer_fini(&feedback_sender->history_buf);
	return err;
}

CHIAKI_EXPORT void chiaki_feedback_sender_fini(ChiakiFeedbackSender *feedback_sender)
{
	chiaki_mutex_lock(&feedback_sender->state_mutex);
	feedback_sender->should_stop = true;
	chiaki_mutex_unlock(&feedback_sender->state_mutex);
	chiaki_cond_signal(&feedback_sender->state_cond);
	chiaki_thread_join(&feedback_sender->thread, NULL);
	feedback_diagnostics_write(feedback_sender->diagnostics);
	free(feedback_sender->diagnostics);
	feedback_sender->diagnostics = NULL;
	chiaki_cond_fini(&feedback_sender->state_cond);
	chiaki_mutex_fini(&feedback_sender->state_mutex);
	chiaki_feedback_history_buffer_fini(&feedback_sender->history_buf);
}

CHIAKI_EXPORT ChiakiErrorCode chiaki_feedback_sender_set_controller_state(ChiakiFeedbackSender *feedback_sender, ChiakiControllerState *state)
{
	uint64_t started = feedback_sender->diagnostics ? feedback_diagnostic_clock() : 0;
	ChiakiErrorCode err = chiaki_mutex_lock(&feedback_sender->state_mutex);
	if(feedback_sender->diagnostics && err == CHIAKI_ERR_SUCCESS)
		feedback_diagnostic_record(feedback_sender->diagnostics, FD_FOREGROUND_WAIT, started, feedback_diagnostic_clock());
	if(err != CHIAKI_ERR_SUCCESS)
		return err;

	if(chiaki_controller_state_equals(&feedback_sender->controller_state, state))
	{
		chiaki_mutex_unlock(&feedback_sender->state_mutex);
		return CHIAKI_ERR_SUCCESS;
	}

	feedback_sender->controller_state = *state;
	feedback_sender_record_history(feedback_sender, &feedback_sender->controller_state_history_prev, &feedback_sender->controller_state);
	feedback_sender->controller_state_history_prev = feedback_sender->controller_state;
	// PS5 Input Bridge: send now on the caller's thread instead of waiting for the sender thread.
	err = feedback_sender_send_now_locked(feedback_sender);
	// Still wake the sender thread: it schedules the tail resends of the packet just sent.
	feedback_sender->controller_state_changed = true;

	chiaki_mutex_unlock(&feedback_sender->state_mutex);
	chiaki_cond_signal(&feedback_sender->state_cond);

	return err;
}

static bool controller_state_equals_for_feedback_state(ChiakiControllerState *a, ChiakiControllerState *b)
{
	if(!(a->left_x == b->left_x
		&& a->left_y == b->left_y
		&& a->right_x == b->right_x
		&& a->right_y == b->right_y))
		return false;
#define CHECKF(n) if(a->n < b->n - 0.0000001f || a->n > b->n + 0.0000001f) return false
	CHECKF(gyro_x);
	CHECKF(gyro_y);
	CHECKF(gyro_z);
	CHECKF(accel_x);
	CHECKF(accel_y);
	CHECKF(accel_z);
	CHECKF(orient_x);
	CHECKF(orient_y);
	CHECKF(orient_z);
	CHECKF(orient_w);
#undef CHECKF
	return true;
}

static ChiakiErrorCode feedback_sender_send_state(ChiakiFeedbackSender *feedback_sender, const ChiakiControllerState *state)
{
	ChiakiFeedbackState feedback_state;
	feedback_state.left_x = state->left_x;
	feedback_state.left_y = state->left_y;
	feedback_state.right_x = state->right_x;
	feedback_state.right_y = state->right_y;
	feedback_state.gyro_x = state->gyro_x;
	feedback_state.gyro_y = state->gyro_y;
	feedback_state.gyro_z = state->gyro_z;
	feedback_state.accel_x = state->accel_x;
	feedback_state.accel_y = state->accel_y;
	feedback_state.accel_z = state->accel_z;

	feedback_state.orient_x = state->orient_x;
	feedback_state.orient_y = state->orient_y;
	feedback_state.orient_z = state->orient_z;
	feedback_state.orient_w = state->orient_w;

	uint64_t started = feedback_sender->diagnostics ? feedback_diagnostic_clock() : 0;
	ChiakiErrorCode err = chiaki_takion_send_feedback_state(feedback_sender->takion, feedback_sender->state_seq_num++, &feedback_state);
	if(feedback_sender->diagnostics)
		feedback_diagnostic_record(feedback_sender->diagnostics, feedback_sender->diagnostics->repeat ? FD_REPEAT_STATE : FD_FOREGROUND_STATE, started, feedback_diagnostic_clock());
	if(err != CHIAKI_ERR_SUCCESS)
		CHIAKI_LOGE(feedback_sender->log, "FeedbackSender failed to send Feedback State");
	return err;
}

static ChiakiErrorCode feedback_sender_send_history_packet(ChiakiFeedbackSender *feedback_sender, uint8_t event_count, const uint8_t *buf, size_t buf_size)
{
	//CHIAKI_LOGD(feedback_sender->log, "Feedback History:");
	//chiaki_log_hexdump(feedback_sender->log, CHIAKI_LOG_DEBUG, buf, buf_size);
	uint64_t started = feedback_sender->diagnostics ? feedback_diagnostic_clock() : 0;
	ChiakiErrorCode err = chiaki_takion_send_feedback_history(feedback_sender->takion, feedback_sender->history_seq_num++, event_count, (uint8_t *)buf, buf_size);
	if(feedback_sender->diagnostics)
		feedback_diagnostic_record(feedback_sender->diagnostics, feedback_sender->diagnostics->repeat ? FD_REPEAT_HISTORY : FD_FOREGROUND_HISTORY, started, feedback_diagnostic_clock());
	return err;
}

static void feedback_sender_flush_history_locked(ChiakiFeedbackSender *feedback_sender)
{
	if(!feedback_sender->history_dirty)
		return;

	size_t packet_index = (feedback_sender->history_packet_begin + feedback_sender->history_packet_len)
		% CHIAKI_FEEDBACK_HISTORY_PACKET_QUEUE_SIZE;
	size_t packet_size = CHIAKI_FEEDBACK_HISTORY_PACKET_BUF_SIZE;
	ChiakiErrorCode err = chiaki_feedback_history_buffer_format(
			&feedback_sender->history_buf,
			feedback_sender->history_packets[packet_index],
			&packet_size);
	if(err != CHIAKI_ERR_SUCCESS)
	{
		CHIAKI_LOGE(feedback_sender->log, "Feedback Sender failed to format history buffer");
		return;
	}
	uint8_t event_count = (uint8_t)feedback_sender->history_buf.len;

	if(feedback_sender->history_packet_len < CHIAKI_FEEDBACK_HISTORY_PACKET_QUEUE_SIZE)
	{
		feedback_sender->history_packet_sizes[packet_index] = packet_size;
		feedback_sender->history_packet_event_counts[packet_index] = event_count;
		feedback_sender->history_packet_len++;
	}
	else
	{
		feedback_sender->history_packet_sizes[feedback_sender->history_packet_begin] = packet_size;
		feedback_sender->history_packet_event_counts[feedback_sender->history_packet_begin] = event_count;
		memcpy(
			feedback_sender->history_packets[feedback_sender->history_packet_begin],
			feedback_sender->history_packets[packet_index],
			packet_size);
		feedback_sender->history_packet_begin = (feedback_sender->history_packet_begin + 1)
			% CHIAKI_FEEDBACK_HISTORY_PACKET_QUEUE_SIZE;
		CHIAKI_LOGW(feedback_sender->log, "Feedback Sender history packet queue overflow");
	}

	if(feedback_sender->history_buf.len > FEEDBACK_HISTORY_RESEND_EVENT_COUNT)
		feedback_sender->history_buf.len = FEEDBACK_HISTORY_RESEND_EVENT_COUNT;
	feedback_sender->history_dirty = false;
}

static void feedback_sender_history_event_pushed(void *user)
{
	ChiakiFeedbackSender *feedback_sender = user;
	feedback_sender->history_dirty = true;
	feedback_sender_flush_history_locked(feedback_sender);
}

static void feedback_sender_record_history(ChiakiFeedbackSender *feedback_sender, const ChiakiControllerState *state_prev, const ChiakiControllerState *state_now)
{
	chiaki_feedback_history_buffer_record_state(&feedback_sender->history_buf, feedback_sender->log, state_prev, state_now,
			feedback_sender_history_event_pushed, feedback_sender); // PS5 Input Bridge: sent by feedback_sender_send_now_locked
}

// PS5 Input Bridge: sends the queued state change on the caller's thread, exactly as the
// sender thread would (one history packet per event, event count in the header), and
// records the last history packet so the sender thread repeats it at the tail offsets.
// Called with state_mutex held; the sender thread also sends with it held.
static ChiakiErrorCode feedback_sender_send_now_locked(ChiakiFeedbackSender *feedback_sender)
{
	ChiakiErrorCode result = CHIAKI_ERR_SUCCESS;
	if(!controller_state_equals_for_feedback_state(&feedback_sender->controller_state, &feedback_sender->controller_state_prev))
	{
		result = feedback_sender_send_state(feedback_sender, &feedback_sender->controller_state);
		feedback_sender->controller_state_prev = feedback_sender->controller_state;
	}

	uint64_t now_ms = chiaki_time_now_monotonic_ms();
	while(feedback_sender->history_packet_len > 0)
	{
		size_t packet_index = feedback_sender->history_packet_begin;
		size_t packet_size = feedback_sender->history_packet_sizes[packet_index];
		uint8_t event_count = feedback_sender->history_packet_event_counts[packet_index];
		ChiakiErrorCode err = feedback_sender_send_history_packet(feedback_sender, event_count,
				feedback_sender->history_packets[packet_index], packet_size);
		if(err != CHIAKI_ERR_SUCCESS)
			result = err;
		memcpy(feedback_sender->history_last_packet, feedback_sender->history_packets[packet_index], packet_size);
		feedback_sender->history_last_size = packet_size;
		feedback_sender->history_last_event_count = event_count;
		feedback_sender->history_last_sent_ms = now_ms;
		feedback_sender->history_tail_resends = 0;
		feedback_sender->history_packet_begin = (packet_index + 1) % CHIAKI_FEEDBACK_HISTORY_PACKET_QUEUE_SIZE;
		feedback_sender->history_packet_len--;
	}
	return result;
}

static bool state_cond_check(void *user)
{
	ChiakiFeedbackSender *feedback_sender = user;
	return feedback_sender->should_stop
		|| feedback_sender->controller_state_changed;
}

static void *feedback_sender_thread_func(void *user)
{
	ChiakiFeedbackSender *feedback_sender = user;
	chiaki_thread_set_affinity(CHIAKI_THREAD_NAME_FEEDBACK);

	ChiakiErrorCode err = chiaki_mutex_lock(&feedback_sender->state_mutex);
	if(err != CHIAKI_ERR_SUCCESS)
		return NULL;

	uint64_t last_feedback_state_ms = chiaki_time_now_monotonic_ms();
	while(true)
	{
		if(feedback_sender->history_packet_len == 0)
		{
			uint64_t now_ms = chiaki_time_now_monotonic_ms();
			uint64_t next_timeout = FEEDBACK_STATE_TIMEOUT_MAX_MS;
			if(now_ms - last_feedback_state_ms < FEEDBACK_STATE_TIMEOUT_MAX_MS)
				next_timeout = FEEDBACK_STATE_TIMEOUT_MAX_MS - (now_ms - last_feedback_state_ms);
			// wake up in time for the next tail resend of the last history packet
			if(feedback_sender->history_tail_resends < FEEDBACK_HISTORY_TAIL_COUNT)
			{
				uint64_t due_ms = feedback_sender->history_last_sent_ms + feedback_history_tail_ms[feedback_sender->history_tail_resends];
				uint64_t until_due = due_ms > now_ms ? due_ms - now_ms : 0;
				if(until_due < next_timeout)
					next_timeout = until_due;
			}

			err = chiaki_cond_timedwait_pred(&feedback_sender->state_cond, &feedback_sender->state_mutex, next_timeout, state_cond_check, feedback_sender);
			if(err != CHIAKI_ERR_SUCCESS && err != CHIAKI_ERR_TIMEOUT)
				break;
		}
		else
		{
			err = CHIAKI_ERR_SUCCESS;
		}

		if(feedback_sender->should_stop)
			break;

		uint64_t diagnostic_started = feedback_sender->diagnostics ? feedback_diagnostic_clock() : 0;
		if(feedback_sender->diagnostics) feedback_sender->diagnostics->repeat = true;
		uint64_t now_ms = chiaki_time_now_monotonic_ms();
		bool send_feedback_state = now_ms - last_feedback_state_ms >= FEEDBACK_STATE_TIMEOUT_MAX_MS;
		ChiakiControllerState state_now = feedback_sender->controller_state;
		bool send_feedback_history = false;
		uint8_t history_buf[CHIAKI_FEEDBACK_HISTORY_PACKET_BUF_SIZE];
		size_t history_buf_size = 0;
		uint8_t history_event_count = 0;

		if(feedback_sender->controller_state_changed)
		{
			// TODO: FEEDBACK_STATE_TIMEOUT_MIN_MS
			feedback_sender->controller_state_changed = false;
			send_feedback_state = true;

			// don't need to send feedback state if nothing relevant changed
			if(controller_state_equals_for_feedback_state(&state_now, &feedback_sender->controller_state_prev))
				send_feedback_state = false;
		} // else: timeout

		if(feedback_sender->history_packet_len > 0)
		{
			size_t packet_index = feedback_sender->history_packet_begin;
			history_buf_size = feedback_sender->history_packet_sizes[packet_index];
			history_event_count = feedback_sender->history_packet_event_counts[packet_index];
			memcpy(history_buf, feedback_sender->history_packets[packet_index], history_buf_size);
			feedback_sender->history_packet_begin = (feedback_sender->history_packet_begin + 1)
				% CHIAKI_FEEDBACK_HISTORY_PACKET_QUEUE_SIZE;
			feedback_sender->history_packet_len--;
			send_feedback_history = true;
			// remember it for tail resends; a new packet restarts the schedule
			memcpy(feedback_sender->history_last_packet, history_buf, history_buf_size);
			feedback_sender->history_last_size = history_buf_size;
			feedback_sender->history_last_event_count = history_event_count;
			feedback_sender->history_last_sent_ms = now_ms;
			feedback_sender->history_tail_resends = 0;
		}
		else if(feedback_sender->history_last_size > 0
			&& feedback_sender->history_tail_resends < FEEDBACK_HISTORY_TAIL_COUNT
			&& now_ms >= feedback_sender->history_last_sent_ms + feedback_history_tail_ms[feedback_sender->history_tail_resends])
		{
			// nothing new since the last history packet: send it again (gets a new sequence number like any packet)
			history_buf_size = feedback_sender->history_last_size;
			history_event_count = feedback_sender->history_last_event_count;
			memcpy(history_buf, feedback_sender->history_last_packet, history_buf_size);
			feedback_sender->history_tail_resends++;
			send_feedback_history = true;
		}
		// PS5 Input Bridge: send with state_mutex held. The input thread sends under it too,
		// so sequence numbers and packets from the two threads never interleave.
		if(send_feedback_state)
			feedback_sender_send_state(feedback_sender, &state_now);

		if(send_feedback_history)
			feedback_sender_send_history_packet(feedback_sender, history_event_count, history_buf, history_buf_size);

		if(send_feedback_state)
		{
			feedback_sender->controller_state_prev = state_now;
			last_feedback_state_ms = chiaki_time_now_monotonic_ms();
		}
		if(feedback_sender->diagnostics) {
			feedback_diagnostic_record(feedback_sender->diagnostics, FD_REPEAT_HOLD, diagnostic_started, feedback_diagnostic_clock());
			feedback_sender->diagnostics->repeat = false;
		}
	}

	chiaki_mutex_unlock(&feedback_sender->state_mutex);

	return NULL;
}
