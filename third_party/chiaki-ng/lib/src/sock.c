// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#include <chiaki/sock.h>
#include <fcntl.h>
#include <string.h>
#ifdef __linux__
#include <pthread.h>
#include <sys/socket.h>
#endif

static char g_bind_iface[16];
#ifdef __linux__
static pthread_mutex_t g_bind_iface_mtx = PTHREAD_MUTEX_INITIALIZER;
#endif

CHIAKI_EXPORT void chiaki_sock_set_bind_iface(const char *iface)
{
#ifdef __linux__
	pthread_mutex_lock(&g_bind_iface_mtx);
#endif
	if(!iface || !iface[0])
	{
		g_bind_iface[0] = '\0';
	}
	else
	{
		size_t n = strlen(iface);
		if(n >= sizeof(g_bind_iface))
			n = sizeof(g_bind_iface) - 1;
		memcpy(g_bind_iface, iface, n);
		g_bind_iface[n] = '\0';
	}
#ifdef __linux__
	pthread_mutex_unlock(&g_bind_iface_mtx);
#endif
}

CHIAKI_EXPORT ChiakiErrorCode chiaki_sock_bind_iface(chiaki_socket_t sock)
{
#ifdef __linux__
	char iface[16];
	pthread_mutex_lock(&g_bind_iface_mtx);
	memcpy(iface, g_bind_iface, sizeof(iface));
	pthread_mutex_unlock(&g_bind_iface_mtx);
	if(!iface[0])
		return CHIAKI_ERR_SUCCESS;
	if(setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, iface, (socklen_t)(strlen(iface) + 1)) < 0)
		return CHIAKI_ERR_NETWORK;
#else
	(void)sock;
#endif
	return CHIAKI_ERR_SUCCESS;
}

CHIAKI_EXPORT ChiakiErrorCode chiaki_socket_set_nonblock(chiaki_socket_t sock, bool nonblock)
{
#ifdef _WIN32
	u_long nbio = nonblock ? 1 : 0;
	if(ioctlsocket(sock, FIONBIO, &nbio) != NO_ERROR)
		return CHIAKI_ERR_UNKNOWN;
#else
	int flags = fcntl(sock, F_GETFL, 0);
	if(flags == -1)
		return CHIAKI_ERR_UNKNOWN;
	flags = nonblock ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
	if(fcntl(sock, F_SETFL, flags) == -1)
		return CHIAKI_ERR_UNKNOWN;
#endif
	return CHIAKI_ERR_SUCCESS;
}