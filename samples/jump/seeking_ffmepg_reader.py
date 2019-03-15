# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:34:12 2018

@author: einfalmo
"""

import skvideo.io

import sys
import os
import stat
import re
import time
import threading
import subprocess as sp
import logging
import json
import warnings

import numpy as np
from decimal import Decimal

from skvideo.io.ffprobe import ffprobe
from skvideo.utils import *
from skvideo import _HAS_FFMPEG
from skvideo import _FFMPEG_PATH
from skvideo import _FFMPEG_SUPPORTED_DECODERS
from skvideo import _FFMPEG_SUPPORTED_ENCODERS
from skvideo import _FFMPEG_APPLICATION


class SeekingFFmpegReader(skvideo.io.FFmpegReader):
    
    def __init__(self, filename, inputdict=None, outputdict=None, verbosity=0):
        """ FFMPEG reader fork with seeking capabilities.

        

        Parameters
        ----------
        filename : string
            Video file path

        inputdict : dict
            Input dictionary parameters, i.e. how to interpret the input file.

        outputdict : dict
            Output dictionary parameters, i.e. how to encode the data 
            when sending back to the python process.

        Returns
        -------
        none

        """
        # check if FFMPEG exists in the path
        assert _HAS_FFMPEG, "Cannot find installation of real FFmpeg (which comes with ffprobe)."

        israw = 0
        self._inputdict = inputdict
        if self._inputdict is None:
            self._inputdict = dict()

        self._outputdict = outputdict
        if self._outputdict is None:
            self._outputdict = dict()
            
        self._verbosity = verbosity

        # General information
        _, self.extension = os.path.splitext(filename)

        # smartphone video data is weird
        self.rotationAngle = '0'

        self.size = os.path.getsize(filename)
        self.probeInfo = ffprobe(filename)

        viddict = {}
        if "video" in self.probeInfo:
            viddict = self.probeInfo["video"]

        self.inputfps = -1
        if ("-r" in self._inputdict):
            self.inputfps = np.int(self._inputdict["-r"])
        elif "@r_frame_rate" in viddict:
            # check for the slash
            frtxt = viddict["@r_frame_rate"]
            parts = frtxt.split('/')
            if len(parts) > 1:
                self.inputfps = np.float(parts[0])/np.float(parts[1])
            else:
                self.inputfps = np.float(frtxt)
        else:
            # simply default to a common 25 fps and warn
            self.inputfps = 25
            # No input frame rate detected. Assuming 25 fps. Consult documentation on I/O if this is not desired.

        # check for transposition tag
        if ('tag' in viddict):
          tagdata = viddict['tag']
          if not isinstance(tagdata, list):
            tagdata = [tagdata]

          for tags in tagdata:
            if tags['@key'] == 'rotate':
              self.rotationAngle = tags['@value']

        # if we don't have width or height at all, raise exception
        if ("-s" in self._inputdict):
            widthheight = self._inputdict["-s"].split('x')
            self.inputwidth = np.int(widthheight[0])
            self.inputheight = np.int(widthheight[1])
        elif (("@width" in viddict) and ("@height" in viddict)):
            self.inputwidth = np.int(viddict["@width"])
            self.inputheight = np.int(viddict["@height"])
        else:
            raise ValueError("No way to determine width or height from video. Need `-s` in `inputdict`. Consult documentation on I/O.")

        # smartphone recordings seem to store data about rotations
        # in tag format. Just swap the width and height
        if self.rotationAngle == '90' or self.rotationAngle == '270':
          self.inputwidth, self.inputheight = self.inputheight, self.inputwidth

        self.bpp = -1 # bits per pixel
        self.pix_fmt = ""
        # completely unsure of this:
        if ("-pix_fmt" in self._inputdict):
            self.pix_fmt = self._inputdict["-pix_fmt"]
        elif ("@pix_fmt" in viddict):
            # parse this bpp
            self.pix_fmt = viddict["@pix_fmt"]
        else:
            self.pix_fmt = "yuvj444p"
            if verbosity != 0:
                warnings.warn("No input color space detected. Assuming yuvj420p.", UserWarning)

        self.inputdepth = np.int(bpplut[self.pix_fmt][0])
        self.bpp = np.int(bpplut[self.pix_fmt][1])

        if (str.encode(self.extension) in [b".raw", b".yuv"]):
            israw = 1

        if ("-vframes" in self._outputdict):
            self.orig_inputframenum = np.int(self._outputdict["-vframes"])
        elif ("@nb_frames" in viddict):
            self.orig_inputframenum = np.int(viddict["@nb_frames"])
        elif israw == 1:
            # we can compute it based on the input size and color space
            self.orig_inputframenum = np.int(self.size / (self.inputwidth * self.inputheight * (self.bpp/8.0)))
        else:
            self.orig_inputframenum = -1


        if israw != 0:
            self._inputdict['-pix_fmt'] = self.pix_fmt
        else:
            # check that the extension makes sense
            assert str.encode(self.extension).lower() in _FFMPEG_SUPPORTED_DECODERS, "Unknown decoder extension: " + self.extension.lower()

        self._filename = filename

        if '-f' not in self._outputdict:
            self._outputdict['-f'] = "image2pipe"

        if '-pix_fmt' not in self._outputdict:
            self._outputdict['-pix_fmt'] = "rgb24"

        if '-s' in self._outputdict:
            widthheight = self._outputdict["-s"].split('x')
            self.outputwidth = np.int(widthheight[0])
            self.outputheight = np.int(widthheight[1])
        else:
            self.outputwidth = self.inputwidth
            self.outputheight = self.inputheight


        self.outputdepth = np.int(bpplut[self._outputdict['-pix_fmt']][0])
        self.outputbpp = np.int(bpplut[self._outputdict['-pix_fmt']][1])

        if '-vcodec' not in self._outputdict:
            self._outputdict['-vcodec'] = "rawvideo"

        if self.orig_inputframenum == -1:
            # open process with supplied arguments,
            # grabbing number of frames using ffprobe
            probecmd = [_FFMPEG_PATH + "/ffprobe"] + ["-v", "error", "-count_frames", "-select_streams", "v:0",
                                                      "-show_entries", "stream=nb_read_frames", "-of",
                                                      "default=nokey=1:noprint_wrappers=1", self._filename]
            self.orig_inputframenum = np.int(check_output(probecmd).decode().split('\n')[0])


        # Create process
        self._proc = None
        self.seek(0)
        

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.stdout.close()
            self._proc.stderr.close()
            self._terminate(0.025)
        self._proc = None
            
            
    def seek(self, frame_index):
        r"""
        Seek to the given frame index. Closes the old frame stream and creates a new one. Afterwards, a new .nextFrame() iterator has to be used.
        If seeking only a short distance, it might be faster to just decode the frames inbetween manually.
        
        Parameters
        ----------
        frame_index : int
            Index of the frame to seek to.

        Returns
        -------
        none
        """
        assert(0 <= frame_index < self.orig_inputframenum)
        self.close()
        frame_duration = Decimal(1) / Decimal(self.inputfps)
        seek_timestamp = frame_duration * Decimal(int(frame_index))
        self.inputframenum = self.orig_inputframenum - frame_index
        
        seek_flags = ["-ss", "{:.3f}".format(seek_timestamp)]
        
        iargs = []
        for key in self._inputdict.keys():
            iargs.append(key)
            iargs.append(self._inputdict[key])
            
        if frame_index > 0:
            iargs += seek_flags

        oargs = []
        for key in self._outputdict.keys():
            oargs.append(key)
            oargs.append(self._outputdict[key])
        

        cmd = [_FFMPEG_PATH + "/" + _FFMPEG_APPLICATION, "-nostats", "-loglevel", "0"] + iargs + ['-i', self._filename] + oargs + ['-']
        if self._verbosity == 1:
            print(cmd)
        self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                              stdout=sp.PIPE, stderr=sp.PIPE)

            
if __name__ == "__main__":
    # Simple seeking test
    # Compares seeked frames with sequentially decoded ones
    vid_file = "/data_ssd/daten/bisp18_schwimmer/Videos/Start/Brandauer/2017-04-25_17-44Brandauer/Rec_2017-04-25_17-44-56_152_248.mp4"
    orig = skvideo.io.FFmpegReader(vid_file)
    orig_stream = orig.nextFrame()
    orig_frames = [next(orig_stream) for i in range(orig.inputframenum)]
    print("Orig loading done")
    seeking = SeekingFFmpegReader(vid_file)
    for i in range(orig.inputframenum):
        seeking.seek(i)
        assert(seeking.inputframenum == orig.inputframenum - i)
        frame = next(seeking.nextFrame())
        assert(np.all(frame == orig_frames[i]))
    orig.close()
    seeking.close()
    
