#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:53:36 2018

@author: einfalmo
"""

import os
import numpy as np
from video_utils.seeking_ffmepg_reader import SeekingFFmpegReader

np.random.seed(0)


class FrameCache():
    
    def __init__(self, size, start_i):
        self._size = size
        self._start_i = 0
        self._end_i = 0
        self._begin = 0
        self._cache = None
        self.reset_cache(start_i)
        
    def shift_cache(self, shift):
        if shift != 0:
            new_start, new_end = self._start_i + shift, self._end_i + shift
            if new_start >= self._end_i or new_end <= self._start_i:
                self.reset_cache(new_start)
            else:
                new_begin = self._begin + shift
                fill_start, fill_end = min(new_begin, self._begin), max(new_begin, self._begin)
                for i in range(fill_start, fill_end):
                    #print i, self._cache_mod(i)
                    self._cache[self._cache_mod(i)] = None
                    
                self._start_i = new_start
                self._end_i = new_end
                self._begin = self._cache_mod(new_begin)
                
    def at(self, index):
        assert(self.index_in_cache(index))
        return self._cache_hit(index)
    
    def set_at(self, index, frame):
        assert(self.index_in_cache(index))
        self._cache[self._cache_mod(self._begin + (index - self._start_i))] = frame
                
    def _cache_mod(self, i):
        return i % self._size
    
    def index_in_cache(self, index):
        return self._start_i <= index < self._end_i
                
    def _cache_hit(self, index):
        if self.index_in_cache(index):
            return self._cache[self._cache_mod(self._begin + (index - self._start_i))]
        else:
            return None
            
    def reset_cache(self, new_start):
        self._start_i = new_start
        self._end_i = new_start + self._size
        self._begin = 0
        self._cache = [None for i in range(self._size)]
        
    def _debug_print(self):
        print("Start: ", self._start_i)
        print("End: ", self._end_i)
        print("Begin: ", self._begin)
        cache_visualization = [None if c is None else "IN" for c in self._cache]
        #print("Cache: ", self._cache)
        print("Cache: ", cache_visualization)
        begin_visualization = ["__" for c in self._cache]
        begin_visualization[self._begin] = "||"
        print("Begin: ", begin_visualization)
        
            
        


class SimpleVideoReader():
    """
    Simple video reader.
    Provides convenient .at() method to access any video frame in the video.
    Uses internal caching mechanism to store recently decoded frames for faster forward/backward stepping.
    """
    
    def __init__(self, video_file, cache_size=100, keyframe_interval=25, seek_threshold=50, frame_offset=0):
        self._video_file = video_file
        self._keyframe_interval=keyframe_interval
        self._seek_threshold = seek_threshold
        assert(frame_offset >= 0)
        self._frame_offset = frame_offset
        assert(os.path.exists(video_file))
        self._video_reader = SeekingFFmpegReader(video_file, verbosity=0)
        self._orig_num_frames = self._video_reader.getShape()[0]
        self.num_frames = self._orig_num_frames - frame_offset
        self._raw_stream = self._video_reader.nextFrame()
        if type(cache_size) is str:
            assert(cache_size == "max")
            real_cache_size = self._orig_num_frames
        else:
            real_cache_size = max(min(cache_size, self.num_frames), 3)
        self._cache = FrameCache(real_cache_size, 0)
        self._last_decoded_i = -1

        
    def at(self, index):
        index += self._frame_offset
        assert(0 <= index < self._orig_num_frames)
        cache_result = None
        if self._cache.index_in_cache(index):    
            cache_result = self._cache.at(index)
        if cache_result is not None:
            return cache_result
        else:   
            result_frame = None
            seek_distance = index - self._last_decoded_i
            if index > self._last_decoded_i:
                if seek_distance <= self._seek_threshold:
                    self._decode_n(seek_distance, shift_cache=True)
                    result_frame = self._cache.at(index)
                    assert(result_frame is not None)
                else:
                    closest_keyframe = (index // self._keyframe_interval) * self._keyframe_interval
                    self._cache.shift_cache(self._sanitize_cache_shift(seek_distance, index))
                    self._video_reader.seek(closest_keyframe)
                    self._raw_stream = self._video_reader.nextFrame()
                    self._last_decoded_i = closest_keyframe - 1
                    self._decode_n(index - self._last_decoded_i, shift_cache=False)
                    result_frame = self._cache.at(index)
                    assert(result_frame is not None)
                    
            elif index < self._last_decoded_i:
                closest_keyframe = (index // self._keyframe_interval) * self._keyframe_interval
                self._cache.shift_cache(self._sanitize_cache_shift(seek_distance, index))
                self._video_reader.seek(closest_keyframe)
                self._raw_stream = self._video_reader.nextFrame()
                self._last_decoded_i = closest_keyframe - 1
                self._decode_n(index - self._last_decoded_i, shift_cache=False)
                result_frame = self._cache.at(index)
                assert(result_frame is not None)
                
            else:
                # should not happen
                assert(False)
                
            return result_frame
        
        
    def _sanitize_cache_shift(self, shift, target_index):
        cache_mid = self._cache._start_i + (self._cache._size // 2)
        true_shift = target_index - cache_mid
        min_cache_mid = self._cache._size // 2
        max_cache_mid = self._orig_num_frames - min_cache_mid
        if true_shift < 0:
            if target_index < min_cache_mid:
                true_shift = true_shift + (min_cache_mid - target_index)
            
        else:
            if target_index > max_cache_mid:
                true_shift = true_shift - (target_index - max_cache_mid)

        return true_shift
        
        
    def _decode_n(self, n, shift_cache=True):
        for i in range(n):
            frame = next(self._raw_stream)
            self._last_decoded_i += 1
            if shift_cache:
                self._cache.shift_cache(self._sanitize_cache_shift(1, self._last_decoded_i))
            if (self._cache.index_in_cache(self._last_decoded_i)):
                self._cache.set_at(self._last_decoded_i, frame)
            
    
    
    def close(self):
        self._video_reader.close()
    
    
if __name__ == "__main__":
        
    vid_file = "/data_ssd/daten/bisp18_schwimmer/Videos/Start/Brandauer/2017-04-25_17-44Brandauer/Rec_2017-04-25_17-44-56_152_248.mp4"
    vr = SimpleVideoReader(vid_file, cache_size=10)
    desired_frame_index = 100
    video_frame = vr.at(desired_frame_index)
    # ... do something with the frame ...
    video_frame2 = vr.at(0)
    # ... do something with the frame ...
    
    # After extracting all needed frames, close the video reader!
    vr.close()
    
    
