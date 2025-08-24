#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define LOG_FILE "/tmp/_chute.log"
#define MAX_LOG_SIZE (200 * 1024 * 1024)
#define MAX_ROTATIONS 4
#define BUFFER_SIZE 8192

static ssize_t (*real_write)(int fd, const void *buf, size_t count) = NULL;
static ssize_t (*real_writev)(int fd, const struct iovec *iov,
                              int iovcnt) = NULL;
static ssize_t (*real_pwrite)(int fd, const void *buf, size_t count,
                              off_t offset) = NULL;
static int (*real_fprintf)(FILE *stream, const char *format, ...) = NULL;
static int (*real_vfprintf)(FILE *stream, const char *format,
                            va_list ap) = NULL;
static int (*real_fputs)(const char *s, FILE *stream) = NULL;
static size_t (*real_fwrite)(const void *ptr, size_t size, size_t nmemb,
                             FILE *stream) = NULL;
static int (*real_puts)(const char *s) = NULL;
static int (*real_putchar)(int c) = NULL;
static int (*real_fputc)(int c, FILE *stream) = NULL;
static int (*real_printf)(const char *format, ...) = NULL;
static int (*real_vprintf)(const char *format, va_list ap) = NULL;

static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
static int log_fd = -1;
static char *program_name = NULL;
static int should_log = 1;

static const char *ignored_programs[] = {"nvidia-smi", "clinfo",   "free",
                                         "top",        "findmnt",  "df",
                                         "top",        "ldconfig", NULL};

static int is_dev_null(int fd) {
  struct stat fd_stat, null_stat;

  if (fstat(fd, &fd_stat) != 0) {
    return 0;
  }

  if (stat("/dev/null", &null_stat) != 0) {
    return 0;
  }

  return (fd_stat.st_dev == null_stat.st_dev &&
          fd_stat.st_ino == null_stat.st_ino);
}

__attribute__((constructor)) static void init(void) {
  real_write = dlsym(RTLD_NEXT, "write");
  real_writev = dlsym(RTLD_NEXT, "writev");
  real_pwrite = dlsym(RTLD_NEXT, "pwrite");
  real_fprintf = dlsym(RTLD_NEXT, "fprintf");
  real_vfprintf = dlsym(RTLD_NEXT, "vfprintf");
  real_fputs = dlsym(RTLD_NEXT, "fputs");
  real_fwrite = dlsym(RTLD_NEXT, "fwrite");
  real_puts = dlsym(RTLD_NEXT, "puts");
  real_putchar = dlsym(RTLD_NEXT, "putchar");
  real_fputc = dlsym(RTLD_NEXT, "fputc");
  real_printf = dlsym(RTLD_NEXT, "printf");
  real_vprintf = dlsym(RTLD_NEXT, "vprintf");

  char path[1024];
  ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (len != -1) {
    path[len] = '\0';
    program_name = strdup(basename(path));

    for (const char **ignored = ignored_programs; *ignored != NULL; ignored++) {
      if (strcmp(program_name, *ignored) == 0) {
        should_log = 0;
        break;
      }
    }
  } else {
    program_name = strdup("unknown");
  }
}

__attribute__((destructor)) static void cleanup(void) {
  if (log_fd >= 0) {
    close(log_fd);
  }
  if (program_name) {
    free(program_name);
  }
}

static void rotate_logs(void) {
  char old_name[256], new_name[256];
  int new_fd;
  snprintf(new_name, sizeof(new_name), "%s.new", LOG_FILE);
  new_fd = open(new_name, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (new_fd < 0)
    return;
  fcntl(new_fd, F_SETFD, FD_CLOEXEC);
  int old_fd = log_fd;
  log_fd = new_fd;
  if (old_fd >= 0) {
    close(old_fd);
  }
  snprintf(old_name, sizeof(old_name), "%s.%d", LOG_FILE, MAX_ROTATIONS);
  unlink(old_name);
  for (int i = MAX_ROTATIONS - 1; i >= 1; i--) {
    snprintf(old_name, sizeof(old_name), "%s.%d", LOG_FILE, i);
    snprintf(new_name, sizeof(new_name), "%s.%d", LOG_FILE, i + 1);
    rename(old_name, new_name);
  }
  snprintf(new_name, sizeof(new_name), "%s.1", LOG_FILE);
  rename(LOG_FILE, new_name);
  snprintf(new_name, sizeof(new_name), "%s.new", LOG_FILE);
  rename(new_name, LOG_FILE);
}

static void open_log(void) {
  if (log_fd < 0) {
    log_fd = open(LOG_FILE, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (log_fd >= 0) {
      fcntl(log_fd, F_SETFD, FD_CLOEXEC);
    }
  }
}

static void get_timestamp(char *buffer, size_t size) {
  time_t now;
  struct tm *tm_info;
  time(&now);
  tm_info = localtime(&now);
  strftime(buffer, size, "%Y-%m-%dT%H:%M:%S", tm_info);
}

static void write_to_log(const void *buf, size_t count) {
  if (count == 0 || !should_log)
    return;
  pthread_mutex_lock(&log_mutex);
  struct stat st;
  if (stat(LOG_FILE, &st) == 0 && st.st_size > MAX_LOG_SIZE) {
    rotate_logs();
  }
  open_log();
  if (log_fd >= 0) {
    char timestamp[64];
    char prefix[256];
    get_timestamp(timestamp, sizeof(timestamp));

    int prefix_len = snprintf(prefix, sizeof(prefix), "%s %d %s: ", timestamp,
                              getpid(), program_name);

    if (real_write(log_fd, prefix, prefix_len) < 0) {
      close(log_fd);
      log_fd = -1;
      open_log();
      if (log_fd >= 0) {
        real_write(log_fd, prefix, prefix_len);
      }
    }
    if (log_fd >= 0) {
      real_write(log_fd, buf, count);
      if (count > 0 && ((char *)buf)[count - 1] != '\n') {
        real_write(log_fd, "\n", 1);
      }
    }
  }

  pthread_mutex_unlock(&log_mutex);
}

ssize_t write(int fd, const void *buf, size_t count) {
  if (!real_write) {
    real_write = dlsym(RTLD_NEXT, "write");
  }

  if (should_log && (fd == STDOUT_FILENO || fd == STDERR_FILENO) &&
      !is_dev_null(fd)) {
    write_to_log(buf, count);
  }

  return real_write(fd, buf, count);
}

ssize_t writev(int fd, const struct iovec *iov, int iovcnt) {
  if (!real_writev) {
    real_writev = dlsym(RTLD_NEXT, "writev");
  }

  if (should_log && (fd == STDOUT_FILENO || fd == STDERR_FILENO) &&
      !is_dev_null(fd)) {
    for (int i = 0; i < iovcnt; i++) {
      write_to_log(iov[i].iov_base, iov[i].iov_len);
    }
  }

  return real_writev(fd, iov, iovcnt);
}

ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset) {
  if (!real_pwrite) {
    real_pwrite = dlsym(RTLD_NEXT, "pwrite");
  }

  if (should_log && (fd == STDOUT_FILENO || fd == STDERR_FILENO) &&
      !is_dev_null(fd)) {
    write_to_log(buf, count);
  }

  return real_pwrite(fd, buf, count, offset);
}

int printf(const char *format, ...) {
  va_list args, args_copy;
  char buffer[BUFFER_SIZE];
  int result;

  if (!real_vprintf) {
    real_vprintf = dlsym(RTLD_NEXT, "vprintf");
  }

  va_start(args, format);

  if (should_log && !is_dev_null(STDOUT_FILENO)) {
    va_copy(args_copy, args);
    vsnprintf(buffer, sizeof(buffer), format, args_copy);
    va_end(args_copy);

    write_to_log(buffer, strlen(buffer));
  }

  result = real_vprintf(format, args);
  va_end(args);

  return result;
}

int vprintf(const char *format, va_list ap) {
  va_list ap_copy;
  char buffer[BUFFER_SIZE];

  if (!real_vprintf) {
    real_vprintf = dlsym(RTLD_NEXT, "vprintf");
  }

  if (should_log && !is_dev_null(STDOUT_FILENO)) {
    va_copy(ap_copy, ap);
    vsnprintf(buffer, sizeof(buffer), format, ap_copy);
    va_end(ap_copy);

    write_to_log(buffer, strlen(buffer));
  }

  return real_vprintf(format, ap);
}

int fprintf(FILE *stream, const char *format, ...) {
  va_list args, args_copy;
  char buffer[BUFFER_SIZE];
  int result;

  if (!real_vfprintf) {
    real_vfprintf = dlsym(RTLD_NEXT, "vfprintf");
  }

  va_start(args, format);

  if (should_log && (stream == stdout || stream == stderr)) {
    int fd = fileno(stream);
    if (!is_dev_null(fd)) {
      va_copy(args_copy, args);
      vsnprintf(buffer, sizeof(buffer), format, args_copy);
      va_end(args_copy);
      write_to_log(buffer, strlen(buffer));
    }
  }

  result = real_vfprintf(stream, format, args);
  va_end(args);

  return result;
}

int vfprintf(FILE *stream, const char *format, va_list ap) {
  va_list ap_copy;
  char buffer[BUFFER_SIZE];

  if (!real_vfprintf) {
    real_vfprintf = dlsym(RTLD_NEXT, "vfprintf");
  }

  if (should_log && (stream == stdout || stream == stderr)) {
    int fd = fileno(stream);
    if (!is_dev_null(fd)) {
      va_copy(ap_copy, ap);
      vsnprintf(buffer, sizeof(buffer), format, ap_copy);
      va_end(ap_copy);
      write_to_log(buffer, strlen(buffer));
    }
  }

  return real_vfprintf(stream, format, ap);
}

int fputs(const char *s, FILE *stream) {
  if (!real_fputs) {
    real_fputs = dlsym(RTLD_NEXT, "fputs");
  }

  if (should_log && (stream == stdout || stream == stderr)) {
    int fd = fileno(stream);
    if (!is_dev_null(fd)) {
      write_to_log(s, strlen(s));
    }
  }

  return real_fputs(s, stream);
}

size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {
  if (!real_fwrite) {
    real_fwrite = dlsym(RTLD_NEXT, "fwrite");
  }

  if (should_log && (stream == stdout || stream == stderr)) {
    int fd = fileno(stream);
    if (!is_dev_null(fd)) {
      write_to_log(ptr, size * nmemb);
    }
  }

  return real_fwrite(ptr, size, nmemb, stream);
}

int puts(const char *s) {
  if (!real_puts) {
    real_puts = dlsym(RTLD_NEXT, "puts");
  }

  if (should_log && !is_dev_null(STDOUT_FILENO)) {
    write_to_log(s, strlen(s));
    write_to_log("\n", 1);
  }

  return real_puts(s);
}

int putchar(int c) {
  if (!real_putchar) {
    real_putchar = dlsym(RTLD_NEXT, "putchar");
  }

  if (should_log && !is_dev_null(STDOUT_FILENO)) {
    char ch = c;
    write_to_log(&ch, 1);
  }

  return real_putchar(c);
}

int fputc(int c, FILE *stream) {
  if (!real_fputc) {
    real_fputc = dlsym(RTLD_NEXT, "fputc");
  }

  if (should_log && (stream == stdout || stream == stderr)) {
    int fd = fileno(stream);
    if (!is_dev_null(fd)) {
      char ch = c;
      write_to_log(&ch, 1);
    }
  }

  return real_fputc(c, stream);
}
