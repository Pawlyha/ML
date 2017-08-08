import threading as thread
import time
import clipboard


class ClipboardDaemon(thread.Thread):
    def __init__(self, id, name, on_clipboard_change):
        super(ClipboardDaemon, self).__init__()
        self.id = id
        self.name = name
        self.onChange = on_clipboard_change
        self.daemon = True

    def run(self):
        print('daemon start')
        prev_value = ''
        while True:
            new_value = clipboard.paste()
            tmp = clipboard.paste()

            if new_value != prev_value and new_value != '':
                self.onChange(new_value)
                prev_value = new_value

            time.sleep(1)
