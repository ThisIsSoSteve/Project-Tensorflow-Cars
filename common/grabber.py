# A port of https://github.com/phoboslab/jsmpeg-vnc/blob/master/source/grabber.c to python
# License information (GPLv3) is here https://github.com/phoboslab/jsmpeg-vnc/blob/master/README.md
from ctypes import Structure, c_int, POINTER, WINFUNCTYPE, windll, WinError, sizeof, c_wchar_p
from ctypes.wintypes import BOOL, HWND, RECT, HDC, HBITMAP, HGDIOBJ, DWORD, LONG, WORD, UINT, LPVOID, LPSTR

import numpy as np

SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000 
DIB_RGB_COLORS = 0
BI_RGB = 0


class BITMAPINFOHEADER(Structure):
    _fields_ = [('biSize', DWORD),
                ('biWidth', LONG),
                ('biHeight', LONG),
                ('biPlanes', WORD),
                ('biBitCount', WORD),
                ('biCompression', DWORD),
                ('biSizeImage', DWORD),
                ('biXPelsPerMeter', LONG),
                ('biYPelsPerMeter', LONG),
                ('biClrUsed', DWORD),
                ('biClrImportant', DWORD)]


def err_on_zero_or_null_check(result, func, args):
    if not result:
        raise WinError()
    return args


def quick_win_define(name, output, *args, **kwargs):
    dllname, fname = name.split('.')
    params = kwargs.get('params', None)
    if params:
        params = tuple([(x, ) for x in params])
    func = (WINFUNCTYPE(output, *args))((fname, getattr(windll, dllname)), params)
    err = kwargs.get('err', err_on_zero_or_null_check)
    if err:
        func.errcheck = err
    return func


GetClientRect = quick_win_define('user32.GetClientRect', BOOL, HWND, POINTER(RECT), params=(1, 2))
GetDC = quick_win_define('user32.GetDC', HDC, HWND)
CreateCompatibleDC = quick_win_define('gdi32.CreateCompatibleDC', HDC, HDC)
CreateCompatibleBitmap = quick_win_define('gdi32.CreateCompatibleBitmap', HBITMAP, HDC, c_int, c_int)
ReleaseDC = quick_win_define('user32.ReleaseDC', c_int, HWND, HDC)
DeleteDC = quick_win_define('gdi32.DeleteDC', BOOL, HDC)
DeleteObject = quick_win_define('gdi32.DeleteObject', BOOL, HGDIOBJ)
SelectObject = quick_win_define('gdi32.SelectObject', HGDIOBJ, HDC, HGDIOBJ)
BitBlt = quick_win_define('gdi32.BitBlt', BOOL, HDC, c_int, c_int, c_int, c_int, HDC, c_int, c_int, DWORD)
GetDIBits = quick_win_define('gdi32.GetDIBits', c_int, HDC, HBITMAP, UINT, UINT, LPVOID, POINTER(BITMAPINFOHEADER), UINT)
GetDesktopWindow = quick_win_define('user32.GetDesktopWindow', HWND)
GetWindowRect = quick_win_define('user32.GetWindowRect', BOOL, HWND, POINTER(RECT), params=(1, 2))
FindWindow = quick_win_define('user32.FindWindowW', HWND, c_wchar_p, c_wchar_p)


class Grabber(object):
    def __init__(self, window_title, with_alpha=False, bbox=None):
        hwnd = FindWindow(None, window_title)

        print(hwnd)
        window = GetDesktopWindow()
        self.window = window

        rect = GetWindowRect(hwnd)

        #for project cars settings #note in game resolution
        #Make sure in display settings the "change the size of text, apps and other items" is 100% -Windows 10
        #Note I'm playing the game in windowed mode at 1920 x 1080
        self.width = 1920
        self.height = 1080

        #Offset, May need to adjust these settings
        self.x = rect.left + 10
        self.y = rect.top + 45


        print('w:{} h:{}'.format(self.width, self.height))

        # if bbox:
        #     bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        #     if not bbox[2] or not bbox[3]:
        #         bbox[2] = self.width - bbox[0]
        #         bbox[3] = self.height - bbox[1]
        #     self.x, self.y, self.width, self.height = bbox
        # else:
        #     self.x = 0
        #     self.y = 0
        self.windowDC = GetDC(window)
        self.memoryDC = CreateCompatibleDC(self.windowDC)
        self.bitmap = CreateCompatibleBitmap(self.windowDC, self.width, self.height)
        self.bitmapInfo = BITMAPINFOHEADER()
        self.bitmapInfo.biSize = sizeof(BITMAPINFOHEADER)
        self.bitmapInfo.biPlanes = 1
        self.bitmapInfo.biBitCount = 32 if with_alpha else 24
        self.bitmapInfo.biWidth = self.width
        self.bitmapInfo.biHeight = -self.height
        self.bitmapInfo.biCompression = BI_RGB
        self.bitmapInfo.biSizeImage = 0
        self.channels = 4 if with_alpha else 3
        self.closed = False

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def close(self):
        if self.closed:
            return
        ReleaseDC(self.window, self.windowDC)
        DeleteDC(self.memoryDC)
        DeleteObject(self.bitmap)
        self.closed = True

    def grab(self, output=None):
        if self.closed:
            raise ValueError('Grabber already closed')
        if output is None:
            output = np.empty((self.height, self.width, self.channels), dtype='uint8')
        else:
            if output.shape != (self.height, self.width, self.channels):
                raise ValueError('Invalid output dimentions')
        SelectObject(self.memoryDC, self.bitmap)
        BitBlt(self.memoryDC, 0, 0, self.width, self.height, self.windowDC, self.x, self.y, SRCCOPY)
        GetDIBits(self.memoryDC, self.bitmap, 0, self.height, output.ctypes.data, self.bitmapInfo, DIB_RGB_COLORS)
        return output


# if __name__ == "__main__":
#    import cv2
#    import ctypes
#    import time
#    from datetime import datetime

#    folder_name = 'F:/Project_Cars_Data/Raw2'

#    time.sleep(5)
#    #from PIL import ImageGrab
#    #bbox=(0, 40, 800, 600) 
#    handle = ctypes.windll.user32.GetForegroundWindow()
#    print(handle)
#    grabber = Grabber(window=handle)

#    for i in range(10):
       
       
#        pic = grabber.grab()

#        #gray_image = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
#        #gray_image = cv2.resize(gray_image, (160,120))
#        save_file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
#        cv2.imwrite(folder_name + '/' + save_file_name + '-image.png', pic)
#        #gray_image = None
#        #pic = None
#        #time.sleep(0.2)
#        #cv2.imshow("image", pic)
#        #cv2.waitKey()
#        #time.sleep(3)

#    #s = time.clock()
#    #for i in range(10):
#    #    pic = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
#    #e = time.clock()
#    #print (e - s) / 10
#    #cv2.imwrite('b.tif', pic)

#    #import ctypes
#    #import time

#    #time.sleep(2)
#    #handle = ctypes.windll.user32.GetForegroundWindow()
#    #print(handle)