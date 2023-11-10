# This is working
import cv2
import filters
from managers import WindowManager, CaptureManager

class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Filtered',
                                            self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)
        
        self._windowManager2 = WindowManager('Un_Filtered',  # CREATING A NEW WINDOW
                                            self.onKeypress)
        self._captureManager2 = CaptureManager(
            cv2.VideoCapture(0), self._windowManager2, True) 
     
        self.keycode = None
        
     

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        self._windowManager2.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame() # CAPTURING A NEW VIDEO
            self._captureManager2.enterFrame() # CAPTURING A NEW VIDEO
            frame = self._captureManager.frame
            frame2 = self._captureManager2.frame
            

            if frame is not None:                

                if self.keycode == 101: # e
                    
                    filters.edgeFilter(frame,frame)
                    
                elif self.keycode == 104: # h 
                    
                    filters.filter_hist(frame,frame)
                
                elif self.keycode == 97: # a
                    filters.adaptive_filter(frame,frame)
                elif self.keycode == 177: #u
                    filters.unsharp_filter(frame,frame)
                elif self.keycode == 115: #s
                    filters.smooth_filter(frame,frame)
   

            self._captureManager.exitFrame()
            self._captureManager2.exitFrame()
            self._windowManager.processEvents()
            self._windowManager2.processEvents()
 

    def onKeypress(self, keycode):

        # STORING THE KEY STROCK
        self.keycode = keycode
        if self.keycode == 27: # escape
            self._windowManager.destroyWindow()
            self._windowManager2.destroyWindow()
            

if __name__=="__main__":
    Cameo().run()