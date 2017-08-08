import clipboard_checker.clipboard_daemon as cld
import threading as tread


def clipboard_changed(new_value):
    print(new_value)


def finish_checking():
    print('daemon exit')
    daemon.join()


daemon = cld.ClipboardDaemon(1, 'clipboard checker', clipboard_changed)
daemon.start()

t = tread.Timer(100.0, finish_checking)
t.start()


