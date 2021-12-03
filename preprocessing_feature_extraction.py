#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8


#insert all necessary libraries

from __future__ import print_function

import pandas as pd
import numpy as np

import os
from os import listdir
from os.path import isfile, join

import IPython.display as ipd

import moviepy.editor as mp
from moviepy.editor import *

import mutagen
from mutagen.mp3 import MP3

import opensmile
import audiofile
import time

from pydub import AudioSegment
from pydub.utils import make_chunks


# In[ ]:


import six
import soundfile
import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim


# In[ ]:


dir_video = r'D:\Data\Videofiles'
dir_audio_train = r'D:\Data\Audiofiles\Train'
dir_audio_test = r'D:\Data\Audiofiles\Test'
dir_chunk = r'D:\Data\Chunked'


# In[ ]:


#create list of all videos and get label for each file
def get_xy(dir_video):
    all_files = os.listdir(os.path.abspath(dir_video))
    labels = []
    for file in all_files:
        startup_name = file.split('_')
        if len(startup_name[1])<=3:
            startup_name = '_'+startup_name[1]+'_'+startup_name[2]
        else:
            startup_name = '_'+startup_name[1]

        df = pd.read_csv('Video details.csv', index_col='datafile', sep = ";")
        timeframe = df.loc[df['File_Name'] == startup_name, ('Invest')]
        invest_indicator = int(timeframe.values[0])
        labels.append(invest_indicator)
    return all_files, labels


# In[ ]:


def convert_pitch_audio(data_part, save_dir):
    for file in data_part:
        startup_name = file.split('_')
        if len(startup_name[1])<=3:
            startup_name = '_'+startup_name[1]+'_'+startup_name[2]
        else:
            startup_name = '_'+startup_name[1]
        #now we have the name of the file
        #let's look for start and end time
        df = pd.read_csv('Video details.csv', index_col='datafile', sep = ";")
        timeframe = df.loc[df['File_Name'] == startup_name, ('Start_Pitch', 'End_Pitch')]
        start_pitch, end_pitch = timeframe.values[0][0], timeframe.values[0][1]
        #let's start converting
        mp4file = dir_video + "\\{0}".format(file)
        mp3file = save_dir + "\\{0}.mp3".format(file[:-4])
        
        VideoClip = VideoFileClip(mp4file)
        VideoClip = VideoClip.subclip(start_pitch, end_pitch)
        AudioClip = VideoClip.audio
        AudioClip.write_audiofile(mp3file, verbose=False, logger = None)
        
        AudioClip.close()
        VideoClip.close()
        print('done with converting: ', file)


# In[ ]:


def creating_chunks(dir_file, file):
    file_name = dir_file + "\\{0}".format(file)
    #find start point form which we take 150 seconds (should be somewhere in the middle to reduce noise)
    audio = MP3(file_name)
    length_in_sec = int(audio.info.length)
    length_video = 150
    
    if length_in_sec < length_video:
        diff = int(length_video-length_in_sec)
        my_audio = AudioFileClip(file_name)
        my_audio2 = AudioFileClip(file_name)
        padding_clip = my_audio2.subclip(0, diff)
        final_clip = concatenate_audioclips([my_audio,padding_clip])
        final_clip.write_audiofile("#temp.mp3", verbose=False, logger=None)
    else: 
        start_point = int((length_in_sec - length_video)/2)
    
        #now we have the start point, let's crop the video from that point to 150 seconds later
        my_audio = AudioFileClip(file_name)
        AudioClip = my_audio.subclip(start_point, (start_point + length_video))
        AudioClip.write_audiofile("#temp.mp3", verbose=False, logger = None)
    
    #now create the chunks
    chunk_audio = AudioSegment.from_mp3("#temp.mp3")
    chunk_length_ms = 2000 # pydub calculates in millisec
    chunks = make_chunks(chunk_audio, chunk_length_ms) # make chunks of 2 seconds
    for i, chunk in enumerate(chunks):
        chunk_file = dir_chunk + "\\chunk_{0}.mp3".format(str(i).zfill(2))
        chunk_name = chunk_file[:-4]+'.wav'
        print("exporting", chunk_name)
        chunk.export(chunk_name, format= "wav")


# In[ ]:


def vggish_features(chunk):
    file_name =  dir_chunk + '\\' + chunk
    
    # In this simple example, we run the examples from a single audio file through
    # the model. If none is provided, we generate a synthetic input.
    examples_batch = vggish_input.wavfile_to_examples(file_name)

    with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
        #print('embedding',embedding_batch)
        #np.save('data_spect.npy',embedding_batch) 
        return embedding_batch


# In[ ]:


def HaF_opensmile(chunk):
    file_name = dir_chunk + '\\' + chunk
    
    signal, sampling_rate = audiofile.read(file_name, always_2d=True)
    
    #set up feature extractor functionals
    smile_func = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        )
    #extract features for the signal; functionals
    #smile_func.feature_names
    df_funct = smile_func.process_signal(signal, sampling_rate)
    df_array_funct = df_funct.iloc[0].to_numpy()
    
    #set up feature extractor lld
    smile_lld = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
    #extract features for the signal; lld
    df_lld = smile_lld.process_signal(signal, sampling_rate)
    df_array_lld = np.mean(df_lld.to_numpy(), axis=0)
    
    HaF_array = np.concatenate((df_array_funct, df_array_lld), axis=0)
    return HaF_array


# In[ ]:




