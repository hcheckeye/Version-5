#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.10),
    on Mon Apr  5 12:34:55 2021
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

import nltk
import nltk


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.2.10'
expName = 'VERSION5'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/hannah_checkeye/Desktop/UMW/Spring 2021/Verbal Shaping Experiment/Version 5/VERSION5_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.DEBUG)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1440, 900], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[1,1,1], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "participantID"
participantIDClock = core.Clock()
explain = visual.TextStim(win=win, name='explain',
    text='In order to use your data, we need to identify you with a unique participant code. \n\nThis code will consist of the following: the first three letters of the street you live on, followed by the last three letters of your mother’s name, followed by the first two letters of your birth month. \n\nFor example, if the street name is ‘Road’, the mother’s name ‘Tracy’, and the birth month is ‘January’, the example code would be ROAACYJA\n\nPlease enter your unique code below and then press enter:',
    font='Arial',
    pos=(0, 0.15), height=0.04, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
participantcode = visual.TextStim(win=win, name='participantcode',
    text=None,
    font='Arial',
    pos=(0, -0.35), height=0.1, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
key_resp_3 = keyboard.Keyboard()

# Initialize components for Routine "instructions"
instructionsClock = core.Clock()
instructions_text = visual.TextStim(win=win, name='instructions_text',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
intruct_resp = keyboard.Keyboard()
text_2 = visual.TextStim(win=win, name='text_2',
    text='Press return to continue',
    font='Arial',
    pos=(0.45, -0.42), height=0.04, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "instructions_practice"
instructions_practiceClock = core.Clock()
practiceinstruct = visual.TextStim(win=win, name='practiceinstruct',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp_2 = keyboard.Keyboard()
return_2 = visual.TextStim(win=win, name='return_2',
    text='Press return to continue',
    font='Arial',
    pos=(0.45, -0.42), height=0.04, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "practice_round"
practice_roundClock = core.Clock()
input_text = visual.TextStim(win=win, name='input_text',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
end = keyboard.Keyboard()
hcountprac = visual.TextStim(win=win, name='hcountprac',
    text='default text',
    font='Arial',
    pos=(0.6, -0.4), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
hcount = 0
displayhcount = visual.TextStim(win=win, name='displayhcount',
    text='Hostage Count:',
    font='Arial',
    pos=(0.4, -0.4), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-4.0);
text_border = visual.Rect(
    win=win, name='text_border',
    width=(0.9, 0.5)[0], height=(0.9, 0.5)[1],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)

# Initialize components for Routine "reinforcement"
reinforcementClock = core.Clock()
hostage_4 = visual.ImageStim(
    win=win,
    name='hostage_4', 
    image='hostage1.png', mask=None,
    ori=0, pos=(0.4, 0), size=(0.3, 0.3),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=0.0)
sound_4 = sound.Sound('raygun.wav', secs=0.5, stereo=True, hamming=True,
    name='sound_4')
sound_4.setVolume(1)
releasenotify = visual.TextStim(win=win, name='releasenotify',
    text='A hostage has been released!',
    font='Arial',
    pos=(-0.2, 0), height=0.1, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "practice_VAT_instructions"
practice_VAT_instructionsClock = core.Clock()
text_14 = visual.TextStim(win=win, name='text_14',
    text='While you are communicating with the officer, a sample fragmented message will be sent to you. You must decode that message, and continue communicating. To decode the message, choose which word best completes the phrase in the middle of the screen by pressing either the left or right arrow key. Please work as quickly as possible. \n\nPress return when you are ready to begin. ',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp_4 = keyboard.Keyboard()

# Initialize components for Routine "VATrest"
VATrestClock = core.Clock()
text_3 = visual.TextStim(win=win, name='text_3',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "practice_VAT"
practice_VATClock = core.Clock()
sentence_3 = visual.TextStim(win=win, name='sentence_3',
    text='default text',
    font='Arial',
    pos=(0, 0.2), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
choice1_3 = visual.TextStim(win=win, name='choice1_3',
    text='default text',
    font='Arial',
    pos=(-0.2, -0.2), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
choice2_3 = visual.TextStim(win=win, name='choice2_3',
    text='default text',
    font='Arial',
    pos=(0.2, -0.2), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
endchoice_3 = keyboard.Keyboard()
VATkeys = visual.TextStim(win=win, name='VATkeys',
    text='Press the left key to choose the left word and the right key to choose the right word',
    font='Arial',
    pos=(0, -0.4), height=0.05, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-4.0);

# Initialize components for Routine "instructions_round1"
instructions_round1Clock = core.Clock()
textinstruct = visual.TextStim(win=win, name='textinstruct',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp = keyboard.Keyboard()
return_3 = visual.TextStim(win=win, name='return_3',
    text='Press return to continue',
    font='Arial',
    pos=(0.45, -0.42), height=0.04, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "conjunctions_text"
conjunctions_textClock = core.Clock()
text_13 = visual.TextStim(win=win, name='text_13',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
end_7 = keyboard.Keyboard()
actualcount1 = visual.TextStim(win=win, name='actualcount1',
    text='default text',
    font='Arial',
    pos=(0.6, -0.4), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-3.0);
displayhcount_7 = visual.TextStim(win=win, name='displayhcount_7',
    text='Hostage Count:',
    font='Arial',
    pos=(0.4, -0.4), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-4.0);
alien = visual.ImageStim(
    win=win,
    name='alien', 
    image='aliens1.png', mask=None,
    ori=0, pos=(-0.6, 0.4), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-5.0)
border_text_4 = visual.Rect(
    win=win, name='border_text_4',
    width=(0.9, 0.5)[0], height=(0.9, 0.5)[1],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=-6.0, interpolate=True)
countdowntimer_4 = visual.TextStim(win=win, name='countdowntimer_4',
    text='default text',
    font='Arial',
    pos=(-0.5, -0.4), height=0.1, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-7.0);

# Initialize components for Routine "reinforcement"
reinforcementClock = core.Clock()
hostage_4 = visual.ImageStim(
    win=win,
    name='hostage_4', 
    image='hostage1.png', mask=None,
    ori=0, pos=(0.4, 0), size=(0.3, 0.3),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=0.0)
sound_4 = sound.Sound('raygun.wav', secs=0.5, stereo=True, hamming=True,
    name='sound_4')
sound_4.setVolume(1)
releasenotify = visual.TextStim(win=win, name='releasenotify',
    text='A hostage has been released!',
    font='Arial',
    pos=(-0.2, 0), height=0.1, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "VAT1instruct"
VAT1instructClock = core.Clock()
instructions_choice1 = visual.TextStim(win=win, name='instructions_choice1',
    text='ALERT: The United States Government has intercepted a message from the Qiczox commander. To decode the message, choose which word best completes the phrase in the middle of the screen by pressing either the left or right arrow key. Please work as quickly as possible. \n\nPress return when you are ready to begin',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endchoice1instruct = keyboard.Keyboard()

# Initialize components for Routine "VATrest"
VATrestClock = core.Clock()
text_3 = visual.TextStim(win=win, name='text_3',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "VAT1"
VAT1Clock = core.Clock()
sentence = visual.TextStim(win=win, name='sentence',
    text='default text',
    font='Arial',
    pos=(0, 0.2), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
choice1 = visual.TextStim(win=win, name='choice1',
    text='default text',
    font='Arial',
    pos=(-0.2, -0.2), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
choice2 = visual.TextStim(win=win, name='choice2',
    text='default text',
    font='Arial',
    pos=(0.2, -0.2), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
image = visual.ImageStim(
    win=win,
    name='image', 
    image='aliens1.png', mask=None,
    ori=0, pos=(-0.6, 0.4), size=(0.3, 0.3),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)
endchoice = keyboard.Keyboard()

# Initialize components for Routine "break_1"
break_1Clock = core.Clock()
break_text = visual.TextStim(win=win, name='break_text',
    text='As the United States Interstellar Messenger and Decoder, we understand how tiresome this job title is. You may take a much-needed break.\n\nIf you wish to skip your break and continue, please press enter. ',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
breaktimer = visual.TextStim(win=win, name='breaktimer',
    text='default text',
    font='Arial',
    pos=(0.6, -0.4), height=0.1, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
endbreak = keyboard.Keyboard()

# Initialize components for Routine "introducecont2"
introducecont2Clock = core.Clock()
instructions_round2 = visual.TextStim(win=win, name='instructions_round2',
    text='You now must resume communication with the alien commanders. You will need to communicate with the commanders by typing on your keyboard. Your messages need to be rapid, complete, and engaging. They should only include letter and punctuation keys, with no special characters, number keys, or function keys. Once you decide that your message is complete, hit the return key to send it and then begin a new message. Remember that these are alien species, so they do not think in the same way you do. \n\nPress return to begin',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endroutine = keyboard.Keyboard()

# Initialize components for Routine "adverbs_text"
adverbs_textClock = core.Clock()
text_12 = visual.TextStim(win=win, name='text_12',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
hcount = 0
countdownStarted = False
end_5 = keyboard.Keyboard()
green_alien_3 = visual.ImageStim(
    win=win,
    name='green_alien_3', 
    image='aliens2.png', mask=None,
    ori=0, pos=(-0.6, 0.4), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)
showtimer_5 = visual.TextStim(win=win, name='showtimer_5',
    text='default text',
    font='Arial',
    pos=(-0.5, -0.4), height=0.1, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-4.0);
actualhcount_3 = visual.TextStim(win=win, name='actualhcount_3',
    text='default text',
    font='Arial',
    pos=(0.6, -0.4), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-5.0);
displayhcount_6 = visual.TextStim(win=win, name='displayhcount_6',
    text='Hostage Count:',
    font='Arial',
    pos=(0.4, -0.4), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-6.0);
textborder_3 = visual.Rect(
    win=win, name='textborder_3',
    width=(0.9, 0.5)[0], height=(0.9, 0.5)[1],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=-7.0, interpolate=True)

# Initialize components for Routine "reinforcement"
reinforcementClock = core.Clock()
hostage_4 = visual.ImageStim(
    win=win,
    name='hostage_4', 
    image='hostage1.png', mask=None,
    ori=0, pos=(0.4, 0), size=(0.3, 0.3),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=0.0)
sound_4 = sound.Sound('raygun.wav', secs=0.5, stereo=True, hamming=True,
    name='sound_4')
sound_4.setVolume(1)
releasenotify = visual.TextStim(win=win, name='releasenotify',
    text='A hostage has been released!',
    font='Arial',
    pos=(-0.2, 0), height=0.1, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "VAT2instruct"
VAT2instructClock = core.Clock()
choice2instructtext = visual.TextStim(win=win, name='choice2instructtext',
    text='ALERT: The United States Government has received another message, but it is from the Kivix commander. To decode the message, choose which word best completes the phrase in the middle of the screen by pressing either the left or right arrow key. Please work as quickly as possible. \n\nPress return to continue',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endchoice2instruct = keyboard.Keyboard()

# Initialize components for Routine "VATrest2"
VATrest2Clock = core.Clock()
text_4 = visual.TextStim(win=win, name='text_4',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "VAT_2"
VAT_2Clock = core.Clock()
sentence_2 = visual.TextStim(win=win, name='sentence_2',
    text='default text',
    font='Arial',
    pos=(0, 0.3), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
choice1_2 = visual.TextStim(win=win, name='choice1_2',
    text='default text',
    font='Arial',
    pos=(-0.2, -0.2), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
choice2_2 = visual.TextStim(win=win, name='choice2_2',
    text='default text',
    font='Arial',
    pos=(0.2, -0.2), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
image_2 = visual.ImageStim(
    win=win,
    name='image_2', 
    image='aliens2.png', mask=None,
    ori=0, pos=(0.5, 0.35), size=(0.3, 0.3),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)
endchoice_2 = keyboard.Keyboard()

# Initialize components for Routine "debrief"
debriefClock = core.Clock()
endexperiment = visual.TextStim(win=win, name='endexperiment',
    text='Thank you for your participation! \n\nPlease fill out the qualtrics survey and review the debriefing form.',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "participantID"-------
continueRoutine = True
# update component parameters for each repeat
modify = False
participantcode.text = ''
event.clearEvents('keyboard')
key_resp_3.keys = []
key_resp_3.rt = []
_key_resp_3_allKeys = []
# keep track of which components have finished
participantIDComponents = [explain, participantcode, key_resp_3]
for thisComponent in participantIDComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
participantIDClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "participantID"-------
while continueRoutine:
    # get current time
    t = participantIDClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=participantIDClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *explain* updates
    if explain.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        explain.frameNStart = frameN  # exact frame index
        explain.tStart = t  # local t and not account for scr refresh
        explain.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(explain, 'tStartRefresh')  # time at next scr refresh
        explain.setAutoDraw(True)
    
    # *participantcode* updates
    if participantcode.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        participantcode.frameNStart = frameN  # exact frame index
        participantcode.tStart = t  # local t and not account for scr refresh
        participantcode.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(participantcode, 'tStartRefresh')  # time at next scr refresh
        participantcode.setAutoDraw(True)
    keys = event.getKeys()
    if len(keys):
        if 'space' in keys:
            participantcode.text = participantcode.text + ' '
        elif 'backspace' in keys:
            participantcode.text = participantcode.text[:-1]
        elif 'lshift' in keys or 'rshift' in keys:
            modify = True
        else:
            if modify:
                participantcode.text = participantcode.text + keys[0].upper()
                modify = False
            else:
                participantcode.text = participantcode.text + keys[0]
                
    
    
    # *key_resp_3* updates
    waitOnFlip = False
    if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_3.frameNStart = frameN  # exact frame index
        key_resp_3.tStart = t  # local t and not account for scr refresh
        key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
        key_resp_3.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_3.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_3.getKeys(keyList=['return'], waitRelease=False)
        _key_resp_3_allKeys.extend(theseKeys)
        if len(_key_resp_3_allKeys):
            key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
            key_resp_3.rt = _key_resp_3_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in participantIDComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "participantID"-------
for thisComponent in participantIDComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('explain.started', explain.tStartRefresh)
thisExp.addData('explain.stopped', explain.tStopRefresh)
thisExp.addData("participant code", participantcode.text)

# the Routine "participantID" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
instructloop = data.TrialHandler(nReps=1, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('instructions.xlsx'),
    seed=None, name='instructloop')
thisExp.addLoop(instructloop)  # add the loop to the experiment
thisInstructloop = instructloop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisInstructloop.rgb)
if thisInstructloop != None:
    for paramName in thisInstructloop:
        exec('{} = thisInstructloop[paramName]'.format(paramName))

for thisInstructloop in instructloop:
    currentLoop = instructloop
    # abbreviate parameter names if possible (e.g. rgb = thisInstructloop.rgb)
    if thisInstructloop != None:
        for paramName in thisInstructloop:
            exec('{} = thisInstructloop[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "instructions"-------
    continueRoutine = True
    # update component parameters for each repeat
    instructions_text.setText(instructionsloop)
    intruct_resp.keys = []
    intruct_resp.rt = []
    _intruct_resp_allKeys = []
    # keep track of which components have finished
    instructionsComponents = [instructions_text, intruct_resp, text_2]
    for thisComponent in instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "instructions"-------
    while continueRoutine:
        # get current time
        t = instructionsClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=instructionsClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_text* updates
        if instructions_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_text.frameNStart = frameN  # exact frame index
            instructions_text.tStart = t  # local t and not account for scr refresh
            instructions_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_text, 'tStartRefresh')  # time at next scr refresh
            instructions_text.setAutoDraw(True)
        
        # *intruct_resp* updates
        waitOnFlip = False
        if intruct_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intruct_resp.frameNStart = frameN  # exact frame index
            intruct_resp.tStart = t  # local t and not account for scr refresh
            intruct_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intruct_resp, 'tStartRefresh')  # time at next scr refresh
            intruct_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intruct_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intruct_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intruct_resp.status == STARTED and not waitOnFlip:
            theseKeys = intruct_resp.getKeys(keyList=['return'], waitRelease=False)
            _intruct_resp_allKeys.extend(theseKeys)
            if len(_intruct_resp_allKeys):
                intruct_resp.keys = _intruct_resp_allKeys[-1].name  # just the last key pressed
                intruct_resp.rt = _intruct_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *text_2* updates
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            text_2.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "instructions"-------
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    instructloop.addData('instructions_text.started', instructions_text.tStartRefresh)
    instructloop.addData('instructions_text.stopped', instructions_text.tStopRefresh)
    # check responses
    if intruct_resp.keys in ['', [], None]:  # No response was made
        intruct_resp.keys = None
    instructloop.addData('intruct_resp.keys',intruct_resp.keys)
    if intruct_resp.keys != None:  # we had a response
        instructloop.addData('intruct_resp.rt', intruct_resp.rt)
    instructloop.addData('intruct_resp.started', intruct_resp.tStartRefresh)
    instructloop.addData('intruct_resp.stopped', intruct_resp.tStopRefresh)
    instructloop.addData('text_2.started', text_2.tStartRefresh)
    instructloop.addData('text_2.stopped', text_2.tStopRefresh)
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed 1 repeats of 'instructloop'


# set up handler to look after randomisation of conditions etc
practice_instructloop = data.TrialHandler(nReps=1, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('instructions_practiceround.xlsx'),
    seed=None, name='practice_instructloop')
thisExp.addLoop(practice_instructloop)  # add the loop to the experiment
thisPractice_instructloop = practice_instructloop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisPractice_instructloop.rgb)
if thisPractice_instructloop != None:
    for paramName in thisPractice_instructloop:
        exec('{} = thisPractice_instructloop[paramName]'.format(paramName))

for thisPractice_instructloop in practice_instructloop:
    currentLoop = practice_instructloop
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_instructloop.rgb)
    if thisPractice_instructloop != None:
        for paramName in thisPractice_instructloop:
            exec('{} = thisPractice_instructloop[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "instructions_practice"-------
    continueRoutine = True
    # update component parameters for each repeat
    practiceinstruct.setText(practice_instructions)
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # keep track of which components have finished
    instructions_practiceComponents = [practiceinstruct, key_resp_2, return_2]
    for thisComponent in instructions_practiceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    instructions_practiceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "instructions_practice"-------
    while continueRoutine:
        # get current time
        t = instructions_practiceClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=instructions_practiceClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practiceinstruct* updates
        if practiceinstruct.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            practiceinstruct.frameNStart = frameN  # exact frame index
            practiceinstruct.tStart = t  # local t and not account for scr refresh
            practiceinstruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practiceinstruct, 'tStartRefresh')  # time at next scr refresh
            practiceinstruct.setAutoDraw(True)
        
        # *key_resp_2* updates
        waitOnFlip = False
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['return'], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *return_2* updates
        if return_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            return_2.frameNStart = frameN  # exact frame index
            return_2.tStart = t  # local t and not account for scr refresh
            return_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(return_2, 'tStartRefresh')  # time at next scr refresh
            return_2.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_practiceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "instructions_practice"-------
    for thisComponent in instructions_practiceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_instructloop.addData('practiceinstruct.started', practiceinstruct.tStartRefresh)
    practice_instructloop.addData('practiceinstruct.stopped', practiceinstruct.tStopRefresh)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    practice_instructloop.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        practice_instructloop.addData('key_resp_2.rt', key_resp_2.rt)
    practice_instructloop.addData('key_resp_2.started', key_resp_2.tStartRefresh)
    practice_instructloop.addData('key_resp_2.stopped', key_resp_2.tStopRefresh)
    practice_instructloop.addData('return_2.started', return_2.tStartRefresh)
    practice_instructloop.addData('return_2.stopped', return_2.tStopRefresh)
    # the Routine "instructions_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed 1 repeats of 'practice_instructloop'


# set up handler to look after randomisation of conditions etc
practice_trials = data.TrialHandler(nReps=4, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='practice_trials')
thisExp.addLoop(practice_trials)  # add the loop to the experiment
thisPractice_trial = practice_trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
if thisPractice_trial != None:
    for paramName in thisPractice_trial:
        exec('{} = thisPractice_trial[paramName]'.format(paramName))

for thisPractice_trial in practice_trials:
    currentLoop = practice_trials
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
    if thisPractice_trial != None:
        for paramName in thisPractice_trial:
            exec('{} = thisPractice_trial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "practice_round"-------
    continueRoutine = True
    # update component parameters for each repeat
    end.keys = []
    end.rt = []
    _end_allKeys = []
    hcountprac.setText(hcount)
    modify = False
    input_text.text = ''
    event.clearEvents('keyboard')
    
    if practice_trials.thisN == 0:
        hcount = 0
    # keep track of which components have finished
    practice_roundComponents = [input_text, end, hcountprac, displayhcount, text_border]
    for thisComponent in practice_roundComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    practice_roundClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "practice_round"-------
    while continueRoutine:
        # get current time
        t = practice_roundClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=practice_roundClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *input_text* updates
        if input_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            input_text.frameNStart = frameN  # exact frame index
            input_text.tStart = t  # local t and not account for scr refresh
            input_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(input_text, 'tStartRefresh')  # time at next scr refresh
            input_text.setAutoDraw(True)
        
        # *end* updates
        waitOnFlip = False
        if end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end.frameNStart = frameN  # exact frame index
            end.tStart = t  # local t and not account for scr refresh
            end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end, 'tStartRefresh')  # time at next scr refresh
            end.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end.status == STARTED and not waitOnFlip:
            theseKeys = end.getKeys(keyList=['return'], waitRelease=False)
            _end_allKeys.extend(theseKeys)
            if len(_end_allKeys):
                end.keys = _end_allKeys[-1].name  # just the last key pressed
                end.rt = _end_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *hcountprac* updates
        if hcountprac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hcountprac.frameNStart = frameN  # exact frame index
            hcountprac.tStart = t  # local t and not account for scr refresh
            hcountprac.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hcountprac, 'tStartRefresh')  # time at next scr refresh
            hcountprac.setAutoDraw(True)
        keys = event.getKeys()
        if len(keys):
            if 'space' in keys:
                input_text.text = input_text.text + ' '
            elif 'backspace' in keys:
                input_text.text = input_text.text[:-1]
            elif 'period' in keys:
                input_text.text = input_text.text + '.'
            elif 'comma' in keys:
                input_text.text = input_text.text + ','
            elif 'apostrophe' in keys:
                input_text.text = input_text.text + '\''
            elif 'question' in keys:
                input_text.text = input_text.text + '?'
            elif 'exclamation' in keys:
                input_text.text = input_text.text + ''
            elif 'at' in keys:
                input_text.text = input_text.text + ''
            elif 'pound' in keys:
                input_text.text = input_text.text + ''
            elif 'dollar' in keys:
                input_text.text = input_text.text + ''
            elif 'percent' in keys:
                input_text.text = input_text.text + ''
            elif 'asciicircum' in keys:
                input_text.text = input_text.text + ''
            elif 'ampersand' in keys:
                input_text.text = input_text.text + ''
            elif 'asterisk' in keys:
                input_text.text = input_text.text + ''
            elif 'parenleft' in keys:
                input_text.text = input_text.text + ''
            elif 'parenright' in keys:
                input_text.text = input_text.text + ''
            elif 'underscore' in keys:
                input_text.text = input_text.text + ''
            elif 'minus' in keys:
                input_text.text = input_text.text + ''
            elif 'equal' in keys:
                input_text.text = input_text.text + ''
            elif 'plus' in keys:
                input_text.text = input_text.text + ''
            elif 'bracketleft' in keys:
                input_text.text = input_text.text + ''
            elif 'bracketright' in keys:
                input_text.text = input_text.text + ''
            elif 'braceleft' in keys:
                input_text.text = input_text.text + ''
            elif 'braceright' in keys:
                input_text.text = input_text.text + ''
            elif 'semicolon' in keys:
                input_text.text = input_text.text + ';'
            elif 'colon' in keys:
                input_text.text = input_text.text + ':'
            elif 'doublequote' in keys:
                input_text.text = input_text.text + ''
            elif 'backslash' in keys:
                input_text.text = input_text.text + ''
            elif 'slash' in keys:
                input_text.text = input_text.text + ''
            elif 'greater' in keys:
                input_text.text = input_text.text + ''
            elif 'less' in keys:
                input_text.text = input_text.text + ''
            elif 'quoteleft' in keys:
                input_text.text = input_text.text + ''
            elif 'asciitilde' in keys:
                input_text.text = input_text.text + ''
            elif 'lshift' in keys or 'rshift' in keys:
                modify = True
            elif 'return' in keys:
                continueRoutine = False
            else:
                if modify:
                    input_text.text = input_text.text + keys[0].upper()
                    modify = False
                else:
                    input_text.text = input_text.text + keys[0]
        
        # *displayhcount* updates
        if displayhcount.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            displayhcount.frameNStart = frameN  # exact frame index
            displayhcount.tStart = t  # local t and not account for scr refresh
            displayhcount.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(displayhcount, 'tStartRefresh')  # time at next scr refresh
            displayhcount.setAutoDraw(True)
        
        # *text_border* updates
        if text_border.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_border.frameNStart = frameN  # exact frame index
            text_border.tStart = t  # local t and not account for scr refresh
            text_border.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_border, 'tStartRefresh')  # time at next scr refresh
            text_border.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in practice_roundComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "practice_round"-------
    for thisComponent in practice_roundComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_trials.addData('input_text.started', input_text.tStartRefresh)
    practice_trials.addData('input_text.stopped', input_text.tStopRefresh)
    # check responses
    if end.keys in ['', [], None]:  # No response was made
        end.keys = None
    practice_trials.addData('end.keys',end.keys)
    if end.keys != None:  # we had a response
        practice_trials.addData('end.rt', end.rt)
    practice_trials.addData('end.started', end.tStartRefresh)
    practice_trials.addData('end.stopped', end.tStopRefresh)
    practice_trials.addData('hcountprac.started', hcountprac.tStartRefresh)
    practice_trials.addData('hcountprac.stopped', hcountprac.tStopRefresh)
    thisExp.addData("practice round input", input_text.text)
    
    if len(input_text.text) >=1:
        nextRoutineNreps = 1
        hcount += 1
    else:
        nextRoutineNreps = 0
    practice_trials.addData('displayhcount.started', displayhcount.tStartRefresh)
    practice_trials.addData('displayhcount.stopped', displayhcount.tStopRefresh)
    practice_trials.addData('text_border.started', text_border.tStartRefresh)
    practice_trials.addData('text_border.stopped', text_border.tStopRefresh)
    # the Routine "practice_round" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    present_reinforcement1 = data.TrialHandler(nReps=nextRoutineNreps, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='present_reinforcement1')
    thisExp.addLoop(present_reinforcement1)  # add the loop to the experiment
    thisPresent_reinforcement1 = present_reinforcement1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPresent_reinforcement1.rgb)
    if thisPresent_reinforcement1 != None:
        for paramName in thisPresent_reinforcement1:
            exec('{} = thisPresent_reinforcement1[paramName]'.format(paramName))
    
    for thisPresent_reinforcement1 in present_reinforcement1:
        currentLoop = present_reinforcement1
        # abbreviate parameter names if possible (e.g. rgb = thisPresent_reinforcement1.rgb)
        if thisPresent_reinforcement1 != None:
            for paramName in thisPresent_reinforcement1:
                exec('{} = thisPresent_reinforcement1[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "reinforcement"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        sound_4.setSound('raygun.wav', secs=0.5, hamming=True)
        sound_4.setVolume(1, log=False)
        # keep track of which components have finished
        reinforcementComponents = [hostage_4, sound_4, releasenotify]
        for thisComponent in reinforcementComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        reinforcementClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "reinforcement"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = reinforcementClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=reinforcementClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *hostage_4* updates
            if hostage_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hostage_4.frameNStart = frameN  # exact frame index
                hostage_4.tStart = t  # local t and not account for scr refresh
                hostage_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hostage_4, 'tStartRefresh')  # time at next scr refresh
                hostage_4.setAutoDraw(True)
            if hostage_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > hostage_4.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    hostage_4.tStop = t  # not accounting for scr refresh
                    hostage_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(hostage_4, 'tStopRefresh')  # time at next scr refresh
                    hostage_4.setAutoDraw(False)
            # start/stop sound_4
            if sound_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_4.frameNStart = frameN  # exact frame index
                sound_4.tStart = t  # local t and not account for scr refresh
                sound_4.tStartRefresh = tThisFlipGlobal  # on global time
                sound_4.play(when=win)  # sync with win flip
            if sound_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_4.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_4.tStop = t  # not accounting for scr refresh
                    sound_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(sound_4, 'tStopRefresh')  # time at next scr refresh
                    sound_4.stop()
            
            # *releasenotify* updates
            if releasenotify.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                releasenotify.frameNStart = frameN  # exact frame index
                releasenotify.tStart = t  # local t and not account for scr refresh
                releasenotify.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(releasenotify, 'tStartRefresh')  # time at next scr refresh
                releasenotify.setAutoDraw(True)
            if releasenotify.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > releasenotify.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    releasenotify.tStop = t  # not accounting for scr refresh
                    releasenotify.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(releasenotify, 'tStopRefresh')  # time at next scr refresh
                    releasenotify.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in reinforcementComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "reinforcement"-------
        for thisComponent in reinforcementComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        present_reinforcement1.addData('hostage_4.started', hostage_4.tStartRefresh)
        present_reinforcement1.addData('hostage_4.stopped', hostage_4.tStopRefresh)
        sound_4.stop()  # ensure sound has stopped at end of routine
        present_reinforcement1.addData('sound_4.started', sound_4.tStartRefresh)
        present_reinforcement1.addData('sound_4.stopped', sound_4.tStopRefresh)
        present_reinforcement1.addData('releasenotify.started', releasenotify.tStartRefresh)
        present_reinforcement1.addData('releasenotify.stopped', releasenotify.tStopRefresh)
    # completed nextRoutineNreps repeats of 'present_reinforcement1'
    
# completed 4 repeats of 'practice_trials'


# ------Prepare to start Routine "practice_VAT_instructions"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_4.keys = []
key_resp_4.rt = []
_key_resp_4_allKeys = []
# keep track of which components have finished
practice_VAT_instructionsComponents = [text_14, key_resp_4]
for thisComponent in practice_VAT_instructionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
practice_VAT_instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "practice_VAT_instructions"-------
while continueRoutine:
    # get current time
    t = practice_VAT_instructionsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=practice_VAT_instructionsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_14* updates
    if text_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_14.frameNStart = frameN  # exact frame index
        text_14.tStart = t  # local t and not account for scr refresh
        text_14.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_14, 'tStartRefresh')  # time at next scr refresh
        text_14.setAutoDraw(True)
    
    # *key_resp_4* updates
    waitOnFlip = False
    if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_4.frameNStart = frameN  # exact frame index
        key_resp_4.tStart = t  # local t and not account for scr refresh
        key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
        key_resp_4.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_4.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_4.getKeys(keyList=['return'], waitRelease=False)
        _key_resp_4_allKeys.extend(theseKeys)
        if len(_key_resp_4_allKeys):
            key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
            key_resp_4.rt = _key_resp_4_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in practice_VAT_instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "practice_VAT_instructions"-------
for thisComponent in practice_VAT_instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_14.started', text_14.tStartRefresh)
thisExp.addData('text_14.stopped', text_14.tStopRefresh)
# check responses
if key_resp_4.keys in ['', [], None]:  # No response was made
    key_resp_4.keys = None
thisExp.addData('key_resp_4.keys',key_resp_4.keys)
if key_resp_4.keys != None:  # we had a response
    thisExp.addData('key_resp_4.rt', key_resp_4.rt)
thisExp.addData('key_resp_4.started', key_resp_4.tStartRefresh)
thisExp.addData('key_resp_4.stopped', key_resp_4.tStopRefresh)
thisExp.nextEntry()
# the Routine "practice_VAT_instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
practice_VAT_trials = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('practicevat.xlsx'),
    seed=None, name='practice_VAT_trials')
thisExp.addLoop(practice_VAT_trials)  # add the loop to the experiment
thisPractice_VAT_trial = practice_VAT_trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisPractice_VAT_trial.rgb)
if thisPractice_VAT_trial != None:
    for paramName in thisPractice_VAT_trial:
        exec('{} = thisPractice_VAT_trial[paramName]'.format(paramName))

for thisPractice_VAT_trial in practice_VAT_trials:
    currentLoop = practice_VAT_trials
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_VAT_trial.rgb)
    if thisPractice_VAT_trial != None:
        for paramName in thisPractice_VAT_trial:
            exec('{} = thisPractice_VAT_trial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "VATrest"-------
    continueRoutine = True
    routineTimer.add(0.500000)
    # update component parameters for each repeat
    # keep track of which components have finished
    VATrestComponents = [text_3]
    for thisComponent in VATrestComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    VATrestClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "VATrest"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = VATrestClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=VATrestClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            text_3.setAutoDraw(True)
        if text_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_3.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                text_3.tStop = t  # not accounting for scr refresh
                text_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_3, 'tStopRefresh')  # time at next scr refresh
                text_3.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in VATrestComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "VATrest"-------
    for thisComponent in VATrestComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_VAT_trials.addData('text_3.started', text_3.tStartRefresh)
    practice_VAT_trials.addData('text_3.stopped', text_3.tStopRefresh)
    
    # ------Prepare to start Routine "practice_VAT"-------
    continueRoutine = True
    # update component parameters for each repeat
    sentence_3.setText(sentences)
    choice1_3.setText(choice_a)
    choice2_3.setText(choice_b)
    endchoice_3.keys = []
    endchoice_3.rt = []
    _endchoice_3_allKeys = []
    # keep track of which components have finished
    practice_VATComponents = [sentence_3, choice1_3, choice2_3, endchoice_3, VATkeys]
    for thisComponent in practice_VATComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    practice_VATClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "practice_VAT"-------
    while continueRoutine:
        # get current time
        t = practice_VATClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=practice_VATClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *sentence_3* updates
        if sentence_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sentence_3.frameNStart = frameN  # exact frame index
            sentence_3.tStart = t  # local t and not account for scr refresh
            sentence_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sentence_3, 'tStartRefresh')  # time at next scr refresh
            sentence_3.setAutoDraw(True)
        
        # *choice1_3* updates
        if choice1_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choice1_3.frameNStart = frameN  # exact frame index
            choice1_3.tStart = t  # local t and not account for scr refresh
            choice1_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choice1_3, 'tStartRefresh')  # time at next scr refresh
            choice1_3.setAutoDraw(True)
        
        # *choice2_3* updates
        if choice2_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choice2_3.frameNStart = frameN  # exact frame index
            choice2_3.tStart = t  # local t and not account for scr refresh
            choice2_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choice2_3, 'tStartRefresh')  # time at next scr refresh
            choice2_3.setAutoDraw(True)
        
        # *endchoice_3* updates
        waitOnFlip = False
        if endchoice_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endchoice_3.frameNStart = frameN  # exact frame index
            endchoice_3.tStart = t  # local t and not account for scr refresh
            endchoice_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endchoice_3, 'tStartRefresh')  # time at next scr refresh
            endchoice_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(endchoice_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(endchoice_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if endchoice_3.status == STARTED and not waitOnFlip:
            theseKeys = endchoice_3.getKeys(keyList=['left', 'right'], waitRelease=False)
            _endchoice_3_allKeys.extend(theseKeys)
            if len(_endchoice_3_allKeys):
                endchoice_3.keys = _endchoice_3_allKeys[-1].name  # just the last key pressed
                endchoice_3.rt = _endchoice_3_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *VATkeys* updates
        if VATkeys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            VATkeys.frameNStart = frameN  # exact frame index
            VATkeys.tStart = t  # local t and not account for scr refresh
            VATkeys.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(VATkeys, 'tStartRefresh')  # time at next scr refresh
            VATkeys.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in practice_VATComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "practice_VAT"-------
    for thisComponent in practice_VATComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_VAT_trials.addData('sentence_3.started', sentence_3.tStartRefresh)
    practice_VAT_trials.addData('sentence_3.stopped', sentence_3.tStopRefresh)
    practice_VAT_trials.addData('choice1_3.started', choice1_3.tStartRefresh)
    practice_VAT_trials.addData('choice1_3.stopped', choice1_3.tStopRefresh)
    practice_VAT_trials.addData('choice2_3.started', choice2_3.tStartRefresh)
    practice_VAT_trials.addData('choice2_3.stopped', choice2_3.tStopRefresh)
    # check responses
    if endchoice_3.keys in ['', [], None]:  # No response was made
        endchoice_3.keys = None
    practice_VAT_trials.addData('endchoice_3.keys',endchoice_3.keys)
    if endchoice_3.keys != None:  # we had a response
        practice_VAT_trials.addData('endchoice_3.rt', endchoice_3.rt)
    practice_VAT_trials.addData('endchoice_3.started', endchoice_3.tStartRefresh)
    practice_VAT_trials.addData('endchoice_3.stopped', endchoice_3.tStopRefresh)
    practice_VAT_trials.addData('VATkeys.started', VATkeys.tStartRefresh)
    practice_VAT_trials.addData('VATkeys.stopped', VATkeys.tStopRefresh)
    # the Routine "practice_VAT" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 1 repeats of 'practice_VAT_trials'


# set up handler to look after randomisation of conditions etc
r1_instruct = data.TrialHandler(nReps=1, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('round1_instructions.xlsx'),
    seed=None, name='r1_instruct')
thisExp.addLoop(r1_instruct)  # add the loop to the experiment
thisR1_instruct = r1_instruct.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisR1_instruct.rgb)
if thisR1_instruct != None:
    for paramName in thisR1_instruct:
        exec('{} = thisR1_instruct[paramName]'.format(paramName))

for thisR1_instruct in r1_instruct:
    currentLoop = r1_instruct
    # abbreviate parameter names if possible (e.g. rgb = thisR1_instruct.rgb)
    if thisR1_instruct != None:
        for paramName in thisR1_instruct:
            exec('{} = thisR1_instruct[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "instructions_round1"-------
    continueRoutine = True
    # update component parameters for each repeat
    textinstruct.setText(instruct_1)
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    instructions_round1Components = [textinstruct, key_resp, return_3]
    for thisComponent in instructions_round1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    instructions_round1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "instructions_round1"-------
    while continueRoutine:
        # get current time
        t = instructions_round1Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=instructions_round1Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textinstruct* updates
        if textinstruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textinstruct.frameNStart = frameN  # exact frame index
            textinstruct.tStart = t  # local t and not account for scr refresh
            textinstruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textinstruct, 'tStartRefresh')  # time at next scr refresh
            textinstruct.setAutoDraw(True)
        
        # *key_resp* updates
        waitOnFlip = False
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['return'], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *return_3* updates
        if return_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            return_3.frameNStart = frameN  # exact frame index
            return_3.tStart = t  # local t and not account for scr refresh
            return_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(return_3, 'tStartRefresh')  # time at next scr refresh
            return_3.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_round1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "instructions_round1"-------
    for thisComponent in instructions_round1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    r1_instruct.addData('textinstruct.started', textinstruct.tStartRefresh)
    r1_instruct.addData('textinstruct.stopped', textinstruct.tStopRefresh)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    r1_instruct.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        r1_instruct.addData('key_resp.rt', key_resp.rt)
    r1_instruct.addData('key_resp.started', key_resp.tStartRefresh)
    r1_instruct.addData('key_resp.stopped', key_resp.tStopRefresh)
    r1_instruct.addData('return_3.started', return_3.tStartRefresh)
    r1_instruct.addData('return_3.stopped', return_3.tStopRefresh)
    # the Routine "instructions_round1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 1 repeats of 'r1_instruct'


# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1000, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "conjunctions_text"-------
    continueRoutine = True
    # update component parameters for each repeat
    modify = False
    text_13.text = ''
    event.clearEvents('keyboard')
    
    if trials.thisN == 0:
        hcount = 0
        
    if not countdownStarted:
        countdownClock = core.CountdownTimer(600) # 300 s = 5 minutes
        countdownStarted = True
    end_7.keys = []
    end_7.rt = []
    _end_7_allKeys = []
    actualcount1.setText(hcount)
    # keep track of which components have finished
    conjunctions_textComponents = [text_13, end_7, actualcount1, displayhcount_7, alien, border_text_4, countdowntimer_4]
    for thisComponent in conjunctions_textComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    conjunctions_textClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "conjunctions_text"-------
    while continueRoutine:
        # get current time
        t = conjunctions_textClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=conjunctions_textClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_13* updates
        if text_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_13.frameNStart = frameN  # exact frame index
            text_13.tStart = t  # local t and not account for scr refresh
            text_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_13, 'tStartRefresh')  # time at next scr refresh
            text_13.setAutoDraw(True)
        keys = event.getKeys()
        if len(keys):
            if 'space' in keys:
                text_13.text = text_13.text + ' '
            elif 'backspace' in keys:
                text_13.text = text_13.text[:-1]
            elif 'period' in keys:
                text_13.text = text_13.text + '.'
            elif 'comma' in keys:
                text_13.text = text_13.text + ','
            elif 'apostrophe' in keys:
                text_13.text = text_13.text + '\''
            elif 'question' in keys:
                text_13.text = text_13.text + '?'
            elif 'exclamation' in keys:
                text_13.text = text_13.text + ''
            elif 'at' in keys:
                text_13.text = text_13.text + ''
            elif 'pound' in keys:
                text_13.text = text_13.text + ''
            elif 'dollar' in keys:
                text_13.text = text_13.text + ''
            elif 'percent' in keys:
                text_13.text = text_13.text + ''
            elif 'asciicircum' in keys:
                text_13.text = text_13.text + ''
            elif 'ampersand' in keys:
                text_13.text = text_13.text + ''
            elif 'asterisk' in keys:
                text_13.text = text_13.text + ''
            elif 'parenleft' in keys:
                text_13.text = text_13.text + ''
            elif 'parenright' in keys:
                text_13.text = text_13.text + ''
            elif 'underscore' in keys:
                text_13.text = text_13.text + ''
            elif 'minus' in keys:
                text_13.text = text_13.text + ''
            elif 'equal' in keys:
                text_13.text = text_13.text + ''
            elif 'plus' in keys:
                text_13.text = text_13.text + ''
            elif 'bracketleft' in keys:
                text_13.text = text_13.text + ''
            elif 'bracketright' in keys:
                text_13.text = text_13.text + ''
            elif 'braceleft' in keys:
                text_13.text = text_13.text + ''
            elif 'braceright' in keys:
                text_13.text = text_13.text + ''
            elif 'semicolon' in keys:
                text_13.text = text_13.text + ';'
            elif 'colon' in keys:
                text_13.text = text_13.text + ':'
            elif 'doublequote' in keys:
                text_13.text = text_13.text + ''
            elif 'backslash' in keys:
                text_13.text = text_13.text + ''
            elif 'slash' in keys:
                text_13.text = text_13.text + ''
            elif 'greater' in keys:
                text_13.text = text_13.text + ''
            elif 'less' in keys:
                text_13.text = text_13.text + ''
            elif 'quoteleft' in keys:
                text_13.text = text_13.text + ''
            elif 'asciitilde' in keys:
                text_13.text = text_13.text + ''
            elif 'lshift' in keys or 'rshift' in keys:
                modify = True
            elif 'return' in keys:
                continueRoutine = False
            else:
                if modify:
                    text_13.text = text_13.text + keys[0].upper()
                    modify = False
                else:
                    text_13.text = text_13.text + keys[0]
                    
            
        timeRemaining = countdownClock.getTime()
        if timeRemaining <= 0.0:
            continueRoutine = False # end this trial immediately
            trials.finished = True # and terminate the loop (use its actual name)
            countdownStarted = False # only necessary if you'll be using the countdown again elsewhere
        else:
            minutes = int(timeRemaining/60.0) # the integer number of minutes
            seconds = int(timeRemaining - (minutes * 60.0))
            timeText = str(minutes) + ':' + str(seconds) # create a string of characters representing the time
        
        # *end_7* updates
        waitOnFlip = False
        if end_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_7.frameNStart = frameN  # exact frame index
            end_7.tStart = t  # local t and not account for scr refresh
            end_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_7, 'tStartRefresh')  # time at next scr refresh
            end_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_7.status == STARTED and not waitOnFlip:
            theseKeys = end_7.getKeys(keyList=['return'], waitRelease=False)
            _end_7_allKeys.extend(theseKeys)
            if len(_end_7_allKeys):
                end_7.keys = [key.name for key in _end_7_allKeys]  # storing all keys
                end_7.rt = [key.rt for key in _end_7_allKeys]
                # a response ends the routine
                continueRoutine = False
        
        # *actualcount1* updates
        if actualcount1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            actualcount1.frameNStart = frameN  # exact frame index
            actualcount1.tStart = t  # local t and not account for scr refresh
            actualcount1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(actualcount1, 'tStartRefresh')  # time at next scr refresh
            actualcount1.setAutoDraw(True)
        
        # *displayhcount_7* updates
        if displayhcount_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            displayhcount_7.frameNStart = frameN  # exact frame index
            displayhcount_7.tStart = t  # local t and not account for scr refresh
            displayhcount_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(displayhcount_7, 'tStartRefresh')  # time at next scr refresh
            displayhcount_7.setAutoDraw(True)
        
        # *alien* updates
        if alien.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            alien.frameNStart = frameN  # exact frame index
            alien.tStart = t  # local t and not account for scr refresh
            alien.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(alien, 'tStartRefresh')  # time at next scr refresh
            alien.setAutoDraw(True)
        
        # *border_text_4* updates
        if border_text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            border_text_4.frameNStart = frameN  # exact frame index
            border_text_4.tStart = t  # local t and not account for scr refresh
            border_text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(border_text_4, 'tStartRefresh')  # time at next scr refresh
            border_text_4.setAutoDraw(True)
        
        # *countdowntimer_4* updates
        if countdowntimer_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            countdowntimer_4.frameNStart = frameN  # exact frame index
            countdowntimer_4.tStart = t  # local t and not account for scr refresh
            countdowntimer_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(countdowntimer_4, 'tStartRefresh')  # time at next scr refresh
            countdowntimer_4.setAutoDraw(True)
        if countdowntimer_4.status == STARTED:  # only update if drawing
            countdowntimer_4.setText(timeText)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in conjunctions_textComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "conjunctions_text"-------
    for thisComponent in conjunctions_textComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('text_13.started', text_13.tStartRefresh)
    trials.addData('text_13.stopped', text_13.tStopRefresh)
    thisExp.addData("tconjunctions input", text_13.text)
    
    displayed_text = text_13.text
    print(displayed_text)
    
    sentences = nltk.word_tokenize(displayed_text)
    words = [nltk.word_tokenize(word) for word in sentences]
    tagged_words = [nltk.pos_tag(sent) for sent in words] #tagged_words is a list of lists of tuples (ordered pairs basically)
    print(tagged_words)
    
    #make an empty freqDist object
    tags = nltk.FreqDist()
            
    #for each list in tagged words, get the (word, tag) tuple, ex. (action, 'NN')
    # 'NN' is the tag in this case
    for pairs in tagged_words:
        for word,tag in pairs:
            #add 1 to the counts for each tag
            tags[tag] += 1
            
    #how to get (and print) the frequency of 'NN's
    print(tags.freq('CC'))
            
    #how to print the FreqDist object
    print(tags.most_common())
            
    #sums up the desired frequencies, k is the 'NN' or 'VRB' or whatever
    freq_sum = 0
    
    for k in tags:
        #if it starts with capital n, its a noun, change this to 'V' if u want verbs
        if(k.startswith('CC')):
            freq_sum += tags.freq(k)
    print("Frequency: ", freq_sum)
    print("Frequency (as %): ", round((freq_sum*(100)),2))
        
    if hcount <= 3 and freq_sum >= 0.1:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <=6 and freq_sum >= 0.15:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <=9 and freq_sum >= 0.2:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <=12 and freq_sum >= 0.25:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <=15 and freq_sum >= 0.3:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 18 and freq_sum >= 0.35:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 21 and freq_sum >= 0.4:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 24 and freq_sum >= 0.45:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 27 and freq_sum >= 0.5:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 30 and freq_sum >= 0.55:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 33 and freq_sum >= 0.6:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 36 and freq_sum >= 0.65:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 39 and freq_sum >= 0.7:
        nextRoutineNreps = 1
        hcount+=1
    else:
        nextRoutineNreps = 0
        
    thisExp.addData('Hostage Count', hcount)
    thisExp.addData('Frequency of conjunctions', freq_sum)
    # check responses
    if end_7.keys in ['', [], None]:  # No response was made
        end_7.keys = None
    trials.addData('end_7.keys',end_7.keys)
    if end_7.keys != None:  # we had a response
        trials.addData('end_7.rt', end_7.rt)
    trials.addData('end_7.started', end_7.tStartRefresh)
    trials.addData('end_7.stopped', end_7.tStopRefresh)
    trials.addData('actualcount1.started', actualcount1.tStartRefresh)
    trials.addData('actualcount1.stopped', actualcount1.tStopRefresh)
    trials.addData('displayhcount_7.started', displayhcount_7.tStartRefresh)
    trials.addData('displayhcount_7.stopped', displayhcount_7.tStopRefresh)
    trials.addData('alien.started', alien.tStartRefresh)
    trials.addData('alien.stopped', alien.tStopRefresh)
    trials.addData('border_text_4.started', border_text_4.tStartRefresh)
    trials.addData('border_text_4.stopped', border_text_4.tStopRefresh)
    trials.addData('countdowntimer_4.started', countdowntimer_4.tStartRefresh)
    trials.addData('countdowntimer_4.stopped', countdowntimer_4.tStopRefresh)
    # the Routine "conjunctions_text" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    present_reinforcement = data.TrialHandler(nReps=nextRoutineNreps, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='present_reinforcement')
    thisExp.addLoop(present_reinforcement)  # add the loop to the experiment
    thisPresent_reinforcement = present_reinforcement.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPresent_reinforcement.rgb)
    if thisPresent_reinforcement != None:
        for paramName in thisPresent_reinforcement:
            exec('{} = thisPresent_reinforcement[paramName]'.format(paramName))
    
    for thisPresent_reinforcement in present_reinforcement:
        currentLoop = present_reinforcement
        # abbreviate parameter names if possible (e.g. rgb = thisPresent_reinforcement.rgb)
        if thisPresent_reinforcement != None:
            for paramName in thisPresent_reinforcement:
                exec('{} = thisPresent_reinforcement[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "reinforcement"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        sound_4.setSound('raygun.wav', secs=0.5, hamming=True)
        sound_4.setVolume(1, log=False)
        # keep track of which components have finished
        reinforcementComponents = [hostage_4, sound_4, releasenotify]
        for thisComponent in reinforcementComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        reinforcementClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "reinforcement"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = reinforcementClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=reinforcementClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *hostage_4* updates
            if hostage_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hostage_4.frameNStart = frameN  # exact frame index
                hostage_4.tStart = t  # local t and not account for scr refresh
                hostage_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hostage_4, 'tStartRefresh')  # time at next scr refresh
                hostage_4.setAutoDraw(True)
            if hostage_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > hostage_4.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    hostage_4.tStop = t  # not accounting for scr refresh
                    hostage_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(hostage_4, 'tStopRefresh')  # time at next scr refresh
                    hostage_4.setAutoDraw(False)
            # start/stop sound_4
            if sound_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_4.frameNStart = frameN  # exact frame index
                sound_4.tStart = t  # local t and not account for scr refresh
                sound_4.tStartRefresh = tThisFlipGlobal  # on global time
                sound_4.play(when=win)  # sync with win flip
            if sound_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_4.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_4.tStop = t  # not accounting for scr refresh
                    sound_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(sound_4, 'tStopRefresh')  # time at next scr refresh
                    sound_4.stop()
            
            # *releasenotify* updates
            if releasenotify.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                releasenotify.frameNStart = frameN  # exact frame index
                releasenotify.tStart = t  # local t and not account for scr refresh
                releasenotify.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(releasenotify, 'tStartRefresh')  # time at next scr refresh
                releasenotify.setAutoDraw(True)
            if releasenotify.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > releasenotify.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    releasenotify.tStop = t  # not accounting for scr refresh
                    releasenotify.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(releasenotify, 'tStopRefresh')  # time at next scr refresh
                    releasenotify.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in reinforcementComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "reinforcement"-------
        for thisComponent in reinforcementComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        present_reinforcement.addData('hostage_4.started', hostage_4.tStartRefresh)
        present_reinforcement.addData('hostage_4.stopped', hostage_4.tStopRefresh)
        sound_4.stop()  # ensure sound has stopped at end of routine
        present_reinforcement.addData('sound_4.started', sound_4.tStartRefresh)
        present_reinforcement.addData('sound_4.stopped', sound_4.tStopRefresh)
        present_reinforcement.addData('releasenotify.started', releasenotify.tStartRefresh)
        present_reinforcement.addData('releasenotify.stopped', releasenotify.tStopRefresh)
    # completed nextRoutineNreps repeats of 'present_reinforcement'
    
# completed 1000 repeats of 'trials'


# ------Prepare to start Routine "VAT1instruct"-------
continueRoutine = True
# update component parameters for each repeat
endchoice1instruct.keys = []
endchoice1instruct.rt = []
_endchoice1instruct_allKeys = []
# keep track of which components have finished
VAT1instructComponents = [instructions_choice1, endchoice1instruct]
for thisComponent in VAT1instructComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
VAT1instructClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "VAT1instruct"-------
while continueRoutine:
    # get current time
    t = VAT1instructClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=VAT1instructClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instructions_choice1* updates
    if instructions_choice1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instructions_choice1.frameNStart = frameN  # exact frame index
        instructions_choice1.tStart = t  # local t and not account for scr refresh
        instructions_choice1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instructions_choice1, 'tStartRefresh')  # time at next scr refresh
        instructions_choice1.setAutoDraw(True)
    
    # *endchoice1instruct* updates
    waitOnFlip = False
    if endchoice1instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endchoice1instruct.frameNStart = frameN  # exact frame index
        endchoice1instruct.tStart = t  # local t and not account for scr refresh
        endchoice1instruct.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endchoice1instruct, 'tStartRefresh')  # time at next scr refresh
        endchoice1instruct.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(endchoice1instruct.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(endchoice1instruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if endchoice1instruct.status == STARTED and not waitOnFlip:
        theseKeys = endchoice1instruct.getKeys(keyList=['return'], waitRelease=False)
        _endchoice1instruct_allKeys.extend(theseKeys)
        if len(_endchoice1instruct_allKeys):
            endchoice1instruct.keys = _endchoice1instruct_allKeys[-1].name  # just the last key pressed
            endchoice1instruct.rt = _endchoice1instruct_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in VAT1instructComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "VAT1instruct"-------
for thisComponent in VAT1instructComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('instructions_choice1.started', instructions_choice1.tStartRefresh)
thisExp.addData('instructions_choice1.stopped', instructions_choice1.tStopRefresh)
# check responses
if endchoice1instruct.keys in ['', [], None]:  # No response was made
    endchoice1instruct.keys = None
thisExp.addData('endchoice1instruct.keys',endchoice1instruct.keys)
if endchoice1instruct.keys != None:  # we had a response
    thisExp.addData('endchoice1instruct.rt', endchoice1instruct.rt)
thisExp.addData('endchoice1instruct.started', endchoice1instruct.tStartRefresh)
thisExp.addData('endchoice1instruct.stopped', endchoice1instruct.tStopRefresh)
thisExp.nextEntry()
# the Routine "VAT1instruct" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
VAT1trials = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('VAT_Conjunction.xlsx'),
    seed=None, name='VAT1trials')
thisExp.addLoop(VAT1trials)  # add the loop to the experiment
thisVAT1trial = VAT1trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisVAT1trial.rgb)
if thisVAT1trial != None:
    for paramName in thisVAT1trial:
        exec('{} = thisVAT1trial[paramName]'.format(paramName))

for thisVAT1trial in VAT1trials:
    currentLoop = VAT1trials
    # abbreviate parameter names if possible (e.g. rgb = thisVAT1trial.rgb)
    if thisVAT1trial != None:
        for paramName in thisVAT1trial:
            exec('{} = thisVAT1trial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "VATrest"-------
    continueRoutine = True
    routineTimer.add(0.500000)
    # update component parameters for each repeat
    # keep track of which components have finished
    VATrestComponents = [text_3]
    for thisComponent in VATrestComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    VATrestClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "VATrest"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = VATrestClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=VATrestClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            text_3.setAutoDraw(True)
        if text_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_3.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                text_3.tStop = t  # not accounting for scr refresh
                text_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_3, 'tStopRefresh')  # time at next scr refresh
                text_3.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in VATrestComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "VATrest"-------
    for thisComponent in VATrestComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    VAT1trials.addData('text_3.started', text_3.tStartRefresh)
    VAT1trials.addData('text_3.stopped', text_3.tStopRefresh)
    
    # ------Prepare to start Routine "VAT1"-------
    continueRoutine = True
    # update component parameters for each repeat
    sentence.setText(sentences)
    choice1.setText(choice_a)
    choice2.setText(choice_b)
    endchoice.keys = []
    endchoice.rt = []
    _endchoice_allKeys = []
    # keep track of which components have finished
    VAT1Components = [sentence, choice1, choice2, image, endchoice]
    for thisComponent in VAT1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    VAT1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "VAT1"-------
    while continueRoutine:
        # get current time
        t = VAT1Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=VAT1Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *sentence* updates
        if sentence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sentence.frameNStart = frameN  # exact frame index
            sentence.tStart = t  # local t and not account for scr refresh
            sentence.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sentence, 'tStartRefresh')  # time at next scr refresh
            sentence.setAutoDraw(True)
        
        # *choice1* updates
        if choice1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choice1.frameNStart = frameN  # exact frame index
            choice1.tStart = t  # local t and not account for scr refresh
            choice1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choice1, 'tStartRefresh')  # time at next scr refresh
            choice1.setAutoDraw(True)
        
        # *choice2* updates
        if choice2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choice2.frameNStart = frameN  # exact frame index
            choice2.tStart = t  # local t and not account for scr refresh
            choice2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choice2, 'tStartRefresh')  # time at next scr refresh
            choice2.setAutoDraw(True)
        
        # *image* updates
        if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image.frameNStart = frameN  # exact frame index
            image.tStart = t  # local t and not account for scr refresh
            image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
            image.setAutoDraw(True)
        
        # *endchoice* updates
        waitOnFlip = False
        if endchoice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endchoice.frameNStart = frameN  # exact frame index
            endchoice.tStart = t  # local t and not account for scr refresh
            endchoice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endchoice, 'tStartRefresh')  # time at next scr refresh
            endchoice.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(endchoice.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(endchoice.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if endchoice.status == STARTED and not waitOnFlip:
            theseKeys = endchoice.getKeys(keyList=['left', 'right'], waitRelease=False)
            _endchoice_allKeys.extend(theseKeys)
            if len(_endchoice_allKeys):
                endchoice.keys = _endchoice_allKeys[-1].name  # just the last key pressed
                endchoice.rt = _endchoice_allKeys[-1].rt
                # was this correct?
                if (endchoice.keys == str(corr_ans)) or (endchoice.keys == corr_ans):
                    endchoice.corr = 1
                else:
                    endchoice.corr = 0
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in VAT1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "VAT1"-------
    for thisComponent in VAT1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    VAT1trials.addData('sentence.started', sentence.tStartRefresh)
    VAT1trials.addData('sentence.stopped', sentence.tStopRefresh)
    VAT1trials.addData('choice1.started', choice1.tStartRefresh)
    VAT1trials.addData('choice1.stopped', choice1.tStopRefresh)
    VAT1trials.addData('choice2.started', choice2.tStartRefresh)
    VAT1trials.addData('choice2.stopped', choice2.tStopRefresh)
    VAT1trials.addData('image.started', image.tStartRefresh)
    VAT1trials.addData('image.stopped', image.tStopRefresh)
    # check responses
    if endchoice.keys in ['', [], None]:  # No response was made
        endchoice.keys = None
        # was no response the correct answer?!
        if str(corr_ans).lower() == 'none':
           endchoice.corr = 1;  # correct non-response
        else:
           endchoice.corr = 0;  # failed to respond (incorrectly)
    # store data for VAT1trials (TrialHandler)
    VAT1trials.addData('endchoice.keys',endchoice.keys)
    VAT1trials.addData('endchoice.corr', endchoice.corr)
    if endchoice.keys != None:  # we had a response
        VAT1trials.addData('endchoice.rt', endchoice.rt)
    VAT1trials.addData('endchoice.started', endchoice.tStartRefresh)
    VAT1trials.addData('endchoice.stopped', endchoice.tStopRefresh)
    # the Routine "VAT1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 1 repeats of 'VAT1trials'


# ------Prepare to start Routine "break_1"-------
continueRoutine = True
routineTimer.add(30.000000)
# update component parameters for each repeat
endbreak.keys = []
endbreak.rt = []
_endbreak_allKeys = []
# keep track of which components have finished
break_1Components = [break_text, breaktimer, endbreak]
for thisComponent in break_1Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
break_1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "break_1"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = break_1Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=break_1Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *break_text* updates
    if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        break_text.frameNStart = frameN  # exact frame index
        break_text.tStart = t  # local t and not account for scr refresh
        break_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
        break_text.setAutoDraw(True)
    if break_text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > break_text.tStartRefresh + 30-frameTolerance:
            # keep track of stop time/frame for later
            break_text.tStop = t  # not accounting for scr refresh
            break_text.frameNStop = frameN  # exact frame index
            win.timeOnFlip(break_text, 'tStopRefresh')  # time at next scr refresh
            break_text.setAutoDraw(False)
    
    # *breaktimer* updates
    if breaktimer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        breaktimer.frameNStart = frameN  # exact frame index
        breaktimer.tStart = t  # local t and not account for scr refresh
        breaktimer.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(breaktimer, 'tStartRefresh')  # time at next scr refresh
        breaktimer.setAutoDraw(True)
    if breaktimer.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > breaktimer.tStartRefresh + 30-frameTolerance:
            # keep track of stop time/frame for later
            breaktimer.tStop = t  # not accounting for scr refresh
            breaktimer.frameNStop = frameN  # exact frame index
            win.timeOnFlip(breaktimer, 'tStopRefresh')  # time at next scr refresh
            breaktimer.setAutoDraw(False)
    if breaktimer.status == STARTED:  # only update if drawing
        breaktimer.setText(round(30.0 - t, ndigits = 1))
    
    # *endbreak* updates
    waitOnFlip = False
    if endbreak.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endbreak.frameNStart = frameN  # exact frame index
        endbreak.tStart = t  # local t and not account for scr refresh
        endbreak.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endbreak, 'tStartRefresh')  # time at next scr refresh
        endbreak.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(endbreak.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(endbreak.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if endbreak.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > endbreak.tStartRefresh + 30-frameTolerance:
            # keep track of stop time/frame for later
            endbreak.tStop = t  # not accounting for scr refresh
            endbreak.frameNStop = frameN  # exact frame index
            win.timeOnFlip(endbreak, 'tStopRefresh')  # time at next scr refresh
            endbreak.status = FINISHED
    if endbreak.status == STARTED and not waitOnFlip:
        theseKeys = endbreak.getKeys(keyList=['return'], waitRelease=False)
        _endbreak_allKeys.extend(theseKeys)
        if len(_endbreak_allKeys):
            endbreak.keys = _endbreak_allKeys[-1].name  # just the last key pressed
            endbreak.rt = _endbreak_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in break_1Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "break_1"-------
for thisComponent in break_1Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('break_text.started', break_text.tStartRefresh)
thisExp.addData('break_text.stopped', break_text.tStopRefresh)
thisExp.addData('breaktimer.started', breaktimer.tStartRefresh)
thisExp.addData('breaktimer.stopped', breaktimer.tStopRefresh)
# check responses
if endbreak.keys in ['', [], None]:  # No response was made
    endbreak.keys = None
thisExp.addData('endbreak.keys',endbreak.keys)
if endbreak.keys != None:  # we had a response
    thisExp.addData('endbreak.rt', endbreak.rt)
thisExp.addData('endbreak.started', endbreak.tStartRefresh)
thisExp.addData('endbreak.stopped', endbreak.tStopRefresh)
thisExp.nextEntry()

# ------Prepare to start Routine "introducecont2"-------
continueRoutine = True
# update component parameters for each repeat
endroutine.keys = []
endroutine.rt = []
_endroutine_allKeys = []
# keep track of which components have finished
introducecont2Components = [instructions_round2, endroutine]
for thisComponent in introducecont2Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
introducecont2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "introducecont2"-------
while continueRoutine:
    # get current time
    t = introducecont2Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=introducecont2Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instructions_round2* updates
    if instructions_round2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instructions_round2.frameNStart = frameN  # exact frame index
        instructions_round2.tStart = t  # local t and not account for scr refresh
        instructions_round2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instructions_round2, 'tStartRefresh')  # time at next scr refresh
        instructions_round2.setAutoDraw(True)
    
    # *endroutine* updates
    waitOnFlip = False
    if endroutine.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endroutine.frameNStart = frameN  # exact frame index
        endroutine.tStart = t  # local t and not account for scr refresh
        endroutine.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endroutine, 'tStartRefresh')  # time at next scr refresh
        endroutine.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(endroutine.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(endroutine.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if endroutine.status == STARTED and not waitOnFlip:
        theseKeys = endroutine.getKeys(keyList=['return'], waitRelease=False)
        _endroutine_allKeys.extend(theseKeys)
        if len(_endroutine_allKeys):
            endroutine.keys = _endroutine_allKeys[-1].name  # just the last key pressed
            endroutine.rt = _endroutine_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in introducecont2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "introducecont2"-------
for thisComponent in introducecont2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('instructions_round2.started', instructions_round2.tStartRefresh)
thisExp.addData('instructions_round2.stopped', instructions_round2.tStopRefresh)
# check responses
if endroutine.keys in ['', [], None]:  # No response was made
    endroutine.keys = None
thisExp.addData('endroutine.keys',endroutine.keys)
if endroutine.keys != None:  # we had a response
    thisExp.addData('endroutine.rt', endroutine.rt)
thisExp.addData('endroutine.started', endroutine.tStartRefresh)
thisExp.addData('endroutine.stopped', endroutine.tStopRefresh)
thisExp.nextEntry()
# the Routine "introducecont2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials_2 = data.TrialHandler(nReps=1000, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials_2')
thisExp.addLoop(trials_2)  # add the loop to the experiment
thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
if thisTrial_2 != None:
    for paramName in thisTrial_2:
        exec('{} = thisTrial_2[paramName]'.format(paramName))

for thisTrial_2 in trials_2:
    currentLoop = trials_2
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            exec('{} = thisTrial_2[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "adverbs_text"-------
    continueRoutine = True
    # update component parameters for each repeat
    modify = False
    text_12.text = ''
    event.clearEvents('keyboard')
    
    if not countdownStarted:
        countdownClock = core.CountdownTimer(600) # 300 s = 5 minutes
        countdownStarted = True
     
    if trials_2.thisN == 0:
        hcount = 0
    end_5.keys = []
    end_5.rt = []
    _end_5_allKeys = []
    actualhcount_3.setText(hcount)
    # keep track of which components have finished
    adverbs_textComponents = [text_12, end_5, green_alien_3, showtimer_5, actualhcount_3, displayhcount_6, textborder_3]
    for thisComponent in adverbs_textComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    adverbs_textClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "adverbs_text"-------
    while continueRoutine:
        # get current time
        t = adverbs_textClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=adverbs_textClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_12* updates
        if text_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_12.frameNStart = frameN  # exact frame index
            text_12.tStart = t  # local t and not account for scr refresh
            text_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_12, 'tStartRefresh')  # time at next scr refresh
            text_12.setAutoDraw(True)
        keys = event.getKeys()
        if len(keys):
            if 'space' in keys:
                text_12.text = text_12.text + ' '
            elif 'backspace' in keys:
                text_12.text = text_12.text[:-1]
            elif 'period' in keys:
                text_12.text = text_12.text + '.'
            elif 'comma' in keys:
                text_12.text = text_12.text + ','
            elif 'apostrophe' in keys:
                text_12.text = text_12.text + '\''
            elif 'question' in keys:
                text_12.text = text_12.text + '?'
            elif 'exclamation' in keys:
                text_12.text = text_12.text + ''
            elif 'at' in keys:
                text_12.text = text_12.text + ''
            elif 'pound' in keys:
                text_12.text = text_12.text + ''
            elif 'dollar' in keys:
                text_12.text = text_12.text + ''
            elif 'percent' in keys:
                text_12.text = text_12.text + ''
            elif 'asciicircum' in keys:
                text_12.text = text_12.text + ''
            elif 'ampersand' in keys:
                text_12.text = text_12.text + ''
            elif 'asterisk' in keys:
                text_12.text = text_12.text + ''
            elif 'parenleft' in keys:
                text_12.text = text_12.text + ''
            elif 'parenright' in keys:
                text_13.text = text_13.text + ''
            elif 'underscore' in keys:
                text_12.text = text_12.text + ''
            elif 'minus' in keys:
                text_12.text = text_12.text + ''
            elif 'equal' in keys:
                text_12.text = text_12.text + ''
            elif 'plus' in keys:
                text_12.text = text_12.text + ''
            elif 'bracketleft' in keys:
                text_12.text = text_12.text + ''
            elif 'bracketright' in keys:
                text_12.text = text_12.text + ''
            elif 'braceleft' in keys:
                text_12.text = text_12.text + ''
            elif 'braceright' in keys:
                text_12.text = text_12.text + ''
            elif 'semicolon' in keys:
                text_12.text = text_12.text + ';'
            elif 'colon' in keys:
                text_12.text = text_12.text + ':'
            elif 'doublequote' in keys:
                text_12.text = text_12.text + ''
            elif 'backslash' in keys:
                text_12.text = text_12.text + ''
            elif 'slash' in keys:
                text_12.text = text_12.text + ''
            elif 'greater' in keys:
                text_12.text = text_12.text + ''
            elif 'less' in keys:
                text_12.text = text_12.text + ''
            elif 'quoteleft' in keys:
                text_12.text = text_12.text + ''
            elif 'asciitilde' in keys:
                text_12.text = text_12.text + ''
            elif 'lshift' in keys or 'rshift' in keys:
                modify = True
            elif 'return' in keys:
                continueRoutine = False
            else:
                if modify:
                    text_12.text = text_12.text + keys[0].upper()
                    modify = False
                else:
                    text_12.text = text_12.text + keys[0]
                    
        timeRemaining = countdownClock.getTime()
        if timeRemaining <= 0.0:
            continueRoutine = False # end this trial immediately
            trials_2.finished = True # and terminate the loop (use its actual name)
            countdownStarted = False # only necessary if you'll be using the countdown again elsewhere
        else:
            minutes = int(timeRemaining/60.0) # the integer number of minutes
            seconds = int(timeRemaining - (minutes * 60.0))
            timeText = str(minutes) + ':' + str(seconds) # create a string of characters representing the time
        
        # *end_5* updates
        waitOnFlip = False
        if end_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_5.frameNStart = frameN  # exact frame index
            end_5.tStart = t  # local t and not account for scr refresh
            end_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_5, 'tStartRefresh')  # time at next scr refresh
            end_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_5.status == STARTED and not waitOnFlip:
            theseKeys = end_5.getKeys(keyList=['return'], waitRelease=False)
            _end_5_allKeys.extend(theseKeys)
            if len(_end_5_allKeys):
                end_5.keys = [key.name for key in _end_5_allKeys]  # storing all keys
                end_5.rt = [key.rt for key in _end_5_allKeys]
                # a response ends the routine
                continueRoutine = False
        
        # *green_alien_3* updates
        if green_alien_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            green_alien_3.frameNStart = frameN  # exact frame index
            green_alien_3.tStart = t  # local t and not account for scr refresh
            green_alien_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(green_alien_3, 'tStartRefresh')  # time at next scr refresh
            green_alien_3.setAutoDraw(True)
        
        # *showtimer_5* updates
        if showtimer_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            showtimer_5.frameNStart = frameN  # exact frame index
            showtimer_5.tStart = t  # local t and not account for scr refresh
            showtimer_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(showtimer_5, 'tStartRefresh')  # time at next scr refresh
            showtimer_5.setAutoDraw(True)
        if showtimer_5.status == STARTED:  # only update if drawing
            showtimer_5.setText(timeText)
        
        # *actualhcount_3* updates
        if actualhcount_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            actualhcount_3.frameNStart = frameN  # exact frame index
            actualhcount_3.tStart = t  # local t and not account for scr refresh
            actualhcount_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(actualhcount_3, 'tStartRefresh')  # time at next scr refresh
            actualhcount_3.setAutoDraw(True)
        
        # *displayhcount_6* updates
        if displayhcount_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            displayhcount_6.frameNStart = frameN  # exact frame index
            displayhcount_6.tStart = t  # local t and not account for scr refresh
            displayhcount_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(displayhcount_6, 'tStartRefresh')  # time at next scr refresh
            displayhcount_6.setAutoDraw(True)
        
        # *textborder_3* updates
        if textborder_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textborder_3.frameNStart = frameN  # exact frame index
            textborder_3.tStart = t  # local t and not account for scr refresh
            textborder_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textborder_3, 'tStartRefresh')  # time at next scr refresh
            textborder_3.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in adverbs_textComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "adverbs_text"-------
    for thisComponent in adverbs_textComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials_2.addData('text_12.started', text_12.tStartRefresh)
    trials_2.addData('text_12.stopped', text_12.tStopRefresh)
    thisExp.addData("adverbs input", text_12.text)
    
    displayed_text = text_12.text
    print(displayed_text)
    
    sentences = nltk.word_tokenize(displayed_text)
    words = [nltk.word_tokenize(word) for word in sentences]
    tagged_words = [nltk.pos_tag(sent) for sent in words] #tagged_words is a list of lists of tuples (ordered pairs basically)
    print(tagged_words)
    
    #make an empty freqDist object
    tags = nltk.FreqDist()
            
    #for each list in tagged words, get the (word, tag) tuple, ex. (action, 'NN')
    # 'NN' is the tag in this case
    for pairs in tagged_words:
        for word,tag in pairs:
            #add 1 to the counts for each tag
            tags[tag] += 1
            
    #how to get (and print) the frequency of 'NN's
    print(tags.freq('RB'))
            
    #how to print the FreqDist object
    print(tags.most_common())
            
    #sums up the desired frequencies, k is the 'NN' or 'VRB' or whatever
    freq_sum = 0
    for k in tags:
        #if it starts with capital n, its a noun, change this to 'V' if u want verbs
        if(k.startswith('RB')):
            freq_sum += tags.freq(k)
    print("Frequency: ", freq_sum)
    print("Frequency (as %): ", round((freq_sum*(100)),2))
        
    if hcount <= 3 and freq_sum >= 0.1:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <=6 and freq_sum >= 0.15:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <=9 and freq_sum >= 0.2:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <=12 and freq_sum >= 0.25:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <=15 and freq_sum >= 0.3:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 18 and freq_sum >= 0.35:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 21 and freq_sum >= 0.4:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 24 and freq_sum >= 0.45:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 27 and freq_sum >= 0.5:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 30 and freq_sum >= 0.55:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 33 and freq_sum >= 0.6:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 36 and freq_sum >= 0.65:
        nextRoutineNreps = 1
        hcount+=1
    elif hcount <= 39 and freq_sum >= 0.7:
        nextRoutineNreps = 1
        hcount+=1
    else:
        nextRoutineNreps = 0
        
    thisExp.addData('Hostage Count', hcount)
    thisExp.addData('Frequency of adverbs', freq_sum)
    # check responses
    if end_5.keys in ['', [], None]:  # No response was made
        end_5.keys = None
    trials_2.addData('end_5.keys',end_5.keys)
    if end_5.keys != None:  # we had a response
        trials_2.addData('end_5.rt', end_5.rt)
    trials_2.addData('end_5.started', end_5.tStartRefresh)
    trials_2.addData('end_5.stopped', end_5.tStopRefresh)
    trials_2.addData('green_alien_3.started', green_alien_3.tStartRefresh)
    trials_2.addData('green_alien_3.stopped', green_alien_3.tStopRefresh)
    trials_2.addData('showtimer_5.started', showtimer_5.tStartRefresh)
    trials_2.addData('showtimer_5.stopped', showtimer_5.tStopRefresh)
    trials_2.addData('actualhcount_3.started', actualhcount_3.tStartRefresh)
    trials_2.addData('actualhcount_3.stopped', actualhcount_3.tStopRefresh)
    trials_2.addData('displayhcount_6.started', displayhcount_6.tStartRefresh)
    trials_2.addData('displayhcount_6.stopped', displayhcount_6.tStopRefresh)
    trials_2.addData('textborder_3.started', textborder_3.tStartRefresh)
    trials_2.addData('textborder_3.stopped', textborder_3.tStopRefresh)
    # the Routine "adverbs_text" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    present_reinforcement2 = data.TrialHandler(nReps=nextRoutineNreps, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='present_reinforcement2')
    thisExp.addLoop(present_reinforcement2)  # add the loop to the experiment
    thisPresent_reinforcement2 = present_reinforcement2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPresent_reinforcement2.rgb)
    if thisPresent_reinforcement2 != None:
        for paramName in thisPresent_reinforcement2:
            exec('{} = thisPresent_reinforcement2[paramName]'.format(paramName))
    
    for thisPresent_reinforcement2 in present_reinforcement2:
        currentLoop = present_reinforcement2
        # abbreviate parameter names if possible (e.g. rgb = thisPresent_reinforcement2.rgb)
        if thisPresent_reinforcement2 != None:
            for paramName in thisPresent_reinforcement2:
                exec('{} = thisPresent_reinforcement2[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "reinforcement"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        sound_4.setSound('raygun.wav', secs=0.5, hamming=True)
        sound_4.setVolume(1, log=False)
        # keep track of which components have finished
        reinforcementComponents = [hostage_4, sound_4, releasenotify]
        for thisComponent in reinforcementComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        reinforcementClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "reinforcement"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = reinforcementClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=reinforcementClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *hostage_4* updates
            if hostage_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hostage_4.frameNStart = frameN  # exact frame index
                hostage_4.tStart = t  # local t and not account for scr refresh
                hostage_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hostage_4, 'tStartRefresh')  # time at next scr refresh
                hostage_4.setAutoDraw(True)
            if hostage_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > hostage_4.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    hostage_4.tStop = t  # not accounting for scr refresh
                    hostage_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(hostage_4, 'tStopRefresh')  # time at next scr refresh
                    hostage_4.setAutoDraw(False)
            # start/stop sound_4
            if sound_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_4.frameNStart = frameN  # exact frame index
                sound_4.tStart = t  # local t and not account for scr refresh
                sound_4.tStartRefresh = tThisFlipGlobal  # on global time
                sound_4.play(when=win)  # sync with win flip
            if sound_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_4.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_4.tStop = t  # not accounting for scr refresh
                    sound_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(sound_4, 'tStopRefresh')  # time at next scr refresh
                    sound_4.stop()
            
            # *releasenotify* updates
            if releasenotify.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                releasenotify.frameNStart = frameN  # exact frame index
                releasenotify.tStart = t  # local t and not account for scr refresh
                releasenotify.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(releasenotify, 'tStartRefresh')  # time at next scr refresh
                releasenotify.setAutoDraw(True)
            if releasenotify.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > releasenotify.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    releasenotify.tStop = t  # not accounting for scr refresh
                    releasenotify.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(releasenotify, 'tStopRefresh')  # time at next scr refresh
                    releasenotify.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in reinforcementComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "reinforcement"-------
        for thisComponent in reinforcementComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        present_reinforcement2.addData('hostage_4.started', hostage_4.tStartRefresh)
        present_reinforcement2.addData('hostage_4.stopped', hostage_4.tStopRefresh)
        sound_4.stop()  # ensure sound has stopped at end of routine
        present_reinforcement2.addData('sound_4.started', sound_4.tStartRefresh)
        present_reinforcement2.addData('sound_4.stopped', sound_4.tStopRefresh)
        present_reinforcement2.addData('releasenotify.started', releasenotify.tStartRefresh)
        present_reinforcement2.addData('releasenotify.stopped', releasenotify.tStopRefresh)
    # completed nextRoutineNreps repeats of 'present_reinforcement2'
    
    thisExp.nextEntry()
    
# completed 1000 repeats of 'trials_2'


# ------Prepare to start Routine "VAT2instruct"-------
continueRoutine = True
# update component parameters for each repeat
endchoice2instruct.keys = []
endchoice2instruct.rt = []
_endchoice2instruct_allKeys = []
# keep track of which components have finished
VAT2instructComponents = [choice2instructtext, endchoice2instruct]
for thisComponent in VAT2instructComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
VAT2instructClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "VAT2instruct"-------
while continueRoutine:
    # get current time
    t = VAT2instructClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=VAT2instructClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *choice2instructtext* updates
    if choice2instructtext.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        choice2instructtext.frameNStart = frameN  # exact frame index
        choice2instructtext.tStart = t  # local t and not account for scr refresh
        choice2instructtext.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(choice2instructtext, 'tStartRefresh')  # time at next scr refresh
        choice2instructtext.setAutoDraw(True)
    
    # *endchoice2instruct* updates
    waitOnFlip = False
    if endchoice2instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endchoice2instruct.frameNStart = frameN  # exact frame index
        endchoice2instruct.tStart = t  # local t and not account for scr refresh
        endchoice2instruct.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endchoice2instruct, 'tStartRefresh')  # time at next scr refresh
        endchoice2instruct.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(endchoice2instruct.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(endchoice2instruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if endchoice2instruct.status == STARTED and not waitOnFlip:
        theseKeys = endchoice2instruct.getKeys(keyList=['return'], waitRelease=False)
        _endchoice2instruct_allKeys.extend(theseKeys)
        if len(_endchoice2instruct_allKeys):
            endchoice2instruct.keys = _endchoice2instruct_allKeys[-1].name  # just the last key pressed
            endchoice2instruct.rt = _endchoice2instruct_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in VAT2instructComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "VAT2instruct"-------
for thisComponent in VAT2instructComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('choice2instructtext.started', choice2instructtext.tStartRefresh)
thisExp.addData('choice2instructtext.stopped', choice2instructtext.tStopRefresh)
# check responses
if endchoice2instruct.keys in ['', [], None]:  # No response was made
    endchoice2instruct.keys = None
thisExp.addData('endchoice2instruct.keys',endchoice2instruct.keys)
if endchoice2instruct.keys != None:  # we had a response
    thisExp.addData('endchoice2instruct.rt', endchoice2instruct.rt)
thisExp.addData('endchoice2instruct.started', endchoice2instruct.tStartRefresh)
thisExp.addData('endchoice2instruct.stopped', endchoice2instruct.tStopRefresh)
thisExp.nextEntry()
# the Routine "VAT2instruct" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
IAT2trials = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('VAT_Adverbs.xlsx'),
    seed=None, name='IAT2trials')
thisExp.addLoop(IAT2trials)  # add the loop to the experiment
thisIAT2trial = IAT2trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisIAT2trial.rgb)
if thisIAT2trial != None:
    for paramName in thisIAT2trial:
        exec('{} = thisIAT2trial[paramName]'.format(paramName))

for thisIAT2trial in IAT2trials:
    currentLoop = IAT2trials
    # abbreviate parameter names if possible (e.g. rgb = thisIAT2trial.rgb)
    if thisIAT2trial != None:
        for paramName in thisIAT2trial:
            exec('{} = thisIAT2trial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "VATrest2"-------
    continueRoutine = True
    routineTimer.add(0.500000)
    # update component parameters for each repeat
    # keep track of which components have finished
    VATrest2Components = [text_4]
    for thisComponent in VATrest2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    VATrest2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "VATrest2"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = VATrest2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=VATrest2Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_4* updates
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            text_4.setAutoDraw(True)
        if text_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_4.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                text_4.tStop = t  # not accounting for scr refresh
                text_4.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_4, 'tStopRefresh')  # time at next scr refresh
                text_4.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in VATrest2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "VATrest2"-------
    for thisComponent in VATrest2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    IAT2trials.addData('text_4.started', text_4.tStartRefresh)
    IAT2trials.addData('text_4.stopped', text_4.tStopRefresh)
    
    # ------Prepare to start Routine "VAT_2"-------
    continueRoutine = True
    # update component parameters for each repeat
    sentence_2.setText(sentences)
    choice1_2.setText(choice_a)
    choice2_2.setText(choice_b)
    endchoice_2.keys = []
    endchoice_2.rt = []
    _endchoice_2_allKeys = []
    # keep track of which components have finished
    VAT_2Components = [sentence_2, choice1_2, choice2_2, image_2, endchoice_2]
    for thisComponent in VAT_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    VAT_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "VAT_2"-------
    while continueRoutine:
        # get current time
        t = VAT_2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=VAT_2Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *sentence_2* updates
        if sentence_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sentence_2.frameNStart = frameN  # exact frame index
            sentence_2.tStart = t  # local t and not account for scr refresh
            sentence_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sentence_2, 'tStartRefresh')  # time at next scr refresh
            sentence_2.setAutoDraw(True)
        
        # *choice1_2* updates
        if choice1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choice1_2.frameNStart = frameN  # exact frame index
            choice1_2.tStart = t  # local t and not account for scr refresh
            choice1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choice1_2, 'tStartRefresh')  # time at next scr refresh
            choice1_2.setAutoDraw(True)
        
        # *choice2_2* updates
        if choice2_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choice2_2.frameNStart = frameN  # exact frame index
            choice2_2.tStart = t  # local t and not account for scr refresh
            choice2_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choice2_2, 'tStartRefresh')  # time at next scr refresh
            choice2_2.setAutoDraw(True)
        
        # *image_2* updates
        if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_2.frameNStart = frameN  # exact frame index
            image_2.tStart = t  # local t and not account for scr refresh
            image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
            image_2.setAutoDraw(True)
        
        # *endchoice_2* updates
        waitOnFlip = False
        if endchoice_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endchoice_2.frameNStart = frameN  # exact frame index
            endchoice_2.tStart = t  # local t and not account for scr refresh
            endchoice_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endchoice_2, 'tStartRefresh')  # time at next scr refresh
            endchoice_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(endchoice_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(endchoice_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if endchoice_2.status == STARTED and not waitOnFlip:
            theseKeys = endchoice_2.getKeys(keyList=['left', 'right'], waitRelease=False)
            _endchoice_2_allKeys.extend(theseKeys)
            if len(_endchoice_2_allKeys):
                endchoice_2.keys = _endchoice_2_allKeys[-1].name  # just the last key pressed
                endchoice_2.rt = _endchoice_2_allKeys[-1].rt
                # was this correct?
                if (endchoice_2.keys == str(corr_ans)) or (endchoice_2.keys == corr_ans):
                    endchoice_2.corr = 1
                else:
                    endchoice_2.corr = 0
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in VAT_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "VAT_2"-------
    for thisComponent in VAT_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    IAT2trials.addData('sentence_2.started', sentence_2.tStartRefresh)
    IAT2trials.addData('sentence_2.stopped', sentence_2.tStopRefresh)
    IAT2trials.addData('choice1_2.started', choice1_2.tStartRefresh)
    IAT2trials.addData('choice1_2.stopped', choice1_2.tStopRefresh)
    IAT2trials.addData('choice2_2.started', choice2_2.tStartRefresh)
    IAT2trials.addData('choice2_2.stopped', choice2_2.tStopRefresh)
    IAT2trials.addData('image_2.started', image_2.tStartRefresh)
    IAT2trials.addData('image_2.stopped', image_2.tStopRefresh)
    # check responses
    if endchoice_2.keys in ['', [], None]:  # No response was made
        endchoice_2.keys = None
        # was no response the correct answer?!
        if str(corr_ans).lower() == 'none':
           endchoice_2.corr = 1;  # correct non-response
        else:
           endchoice_2.corr = 0;  # failed to respond (incorrectly)
    # store data for IAT2trials (TrialHandler)
    IAT2trials.addData('endchoice_2.keys',endchoice_2.keys)
    IAT2trials.addData('endchoice_2.corr', endchoice_2.corr)
    if endchoice_2.keys != None:  # we had a response
        IAT2trials.addData('endchoice_2.rt', endchoice_2.rt)
    IAT2trials.addData('endchoice_2.started', endchoice_2.tStartRefresh)
    IAT2trials.addData('endchoice_2.stopped', endchoice_2.tStopRefresh)
    # the Routine "VAT_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 1 repeats of 'IAT2trials'


# ------Prepare to start Routine "debrief"-------
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
debriefComponents = [endexperiment]
for thisComponent in debriefComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
debriefClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "debrief"-------
while continueRoutine:
    # get current time
    t = debriefClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=debriefClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *endexperiment* updates
    if endexperiment.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endexperiment.frameNStart = frameN  # exact frame index
        endexperiment.tStart = t  # local t and not account for scr refresh
        endexperiment.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endexperiment, 'tStartRefresh')  # time at next scr refresh
        endexperiment.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in debriefComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "debrief"-------
for thisComponent in debriefComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('endexperiment.started', endexperiment.tStartRefresh)
thisExp.addData('endexperiment.stopped', endexperiment.tStopRefresh)
# the Routine "debrief" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
