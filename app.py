from __future__ import print_function

import sklearn

import media as media

import flask
from flask_mysqldb import MySQL
from functools32 import wraps
from passlib.handlers.sha2_crypt import sha256_crypt

from DBconnect import connection
from IPython.core.display import display
from flask import Flask, render_template, request, jsonify, json, session, escape, redirect, url_for, flash
from hashlib import md5
import gc
import requests
import dt
from flask_wtf import Form
from sqlalchemy.dialects import mysql
from werkzeug.utils import secure_filename
from werkzeug.wrappers import Response
from wtforms import BooleanField, TextField, PasswordField, DateField, validators
import sys
import datetime
import time

import pandas as pd
import math
import numpy
from scipy.stats import skew, kurtosis
from statsmodels.tsa import stattools
import operator

import os
import csv
from cStringIO import StringIO
import glob

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioTrainTest as aT
from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks

from ftplib import FTP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.externals import joblib

app = Flask(__name__)
app.debug = True
app.secret_key = 'quiet!'
coding='utf-8'
myPath = ""

@app.route("/")
def main():
    #aT.featureAndTrain(["/home/ahmed/Desktop/MasterThesisTherapio/Datasets/depressed_speech","/home/ahmed/Desktop/MasterThesisTherapio/Datasets/normal speech samples/normal_speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmDepNormExtended")
    #aT.featureAndTrain(["/home/ahmed/Desktop/MasterThesisTherapio/Datasets/depressed_speech", "/home/ahmed/Desktop/MasterThesisTherapio/Datasets/normal_speech_samples/normal_speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "supportvectorNormDepExtended")
    #aT.featureAndTrain(["/home/ahmed/Desktop/MasterThesisTherapio/Datasets/depressed_speech", "/home/ahmed/Desktop/MasterThesisTherapio/Datasets/normal_speech_samples/normal_speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "randomNormDepExtended")
    #aT.featureAndTrain(["/home/ahmed/Desktop/MasterThesisTherapio/Datasets/depressed_speech", "/home/ahmed/Desktop/MasterThesisTherapio/Datasets/normal_speech_samples/normal_speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "extratrees", "xTreesNormDepExtended")
    #aT.featureAndTrain(["/home/ahmed/Desktop/MasterThesisTherapio/Datasets/depressed_speech", "/home/ahmed/Desktop/MasterThesisTherapio/Datasets/normal_speech_samples/normal_speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knearestNormDepExtended")
    #aT.featureAndTrain(["/home/ahmed/Desktop/MasterThesisTherapio/Datasets/depressed_speech", "/home/ahmed/Desktop/MasterThesisTherapio/Datasets/normal_speech_samples/normal_speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "gradientboosting", "gradBoostNormDepExtended")

    #print(aT.fileClassification("/home/ahmed/Desktop/MasterThesisTherapio/Datasets/depressed_speech/depressedmalechange2.wav", "/home/ahmed/RFNormDep", "randomforest"))

    if 'username' in session:
        return render_template('index.html', session_user_name=session['username'])
    return redirect(url_for('showSignIn'))

""" Authentication module """
@app.route("/showSignUp", methods=["GET", "POST"])
def showSignUp():
    try:
        if request.method == "POST":
            docname = request.form['inputName']
            print(docname)
            email = request.form['inputEmail']
            print(email)
            password = sha256_crypt.encrypt((str(request.form['inputPassword'])))
            print(password)
            today = datetime.datetime.today().strftime("%m/%d/%Y")

            c, conn = connection()

            x = c.execute("SELECT * FROM doctor WHERE docname = (%s)", docname)
            print (x)
            print ("executed")
            if int(x) > 0:
                print ("x is bigger than zero")
                flash("That username is already taken, please choose another")
                return render_template('signup.html')

            else:
                c.execute("""INSERT INTO doctor
                                              (doc_id,
                                              docname,
                                              email,
                                              password,
                                              created_at)
                                              VALUES (%s,
                                                      %s,
                                                      %s,
                                                      %s,
                                                      %s)""", (docname,
                                                               docname,
                                                               email,
                                                               password,
                                                               today))

                conn.commit()

                flash("Thanks for registering!")
                c.close()
                conn.close()
                gc.collect()

                session['logged_in'] = True
                session['username'] = docname

                return redirect(url_for('main'))

        return render_template("signup.html")

    except Exception as e:
        return (str(e))
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash("You need to login first")
            return redirect(url_for('login_page'))
    return wrap
@app.route("/showSignIn", methods=["GET", "POST"])
def showSignIn():
    error = ''
    try:
        c, conn = connection()
        if request.method == "POST":
            inputname = request.form['inputName']

            data = c.execute("SELECT * FROM doctor WHERE docname = (%s)",
                             [inputname])

            data = c.fetchone()[4]

            if sha256_crypt.verify(request.form['inputPassword'], data):
                session['logged_in'] = True
                session['username'] = request.form['inputName']

                flash("You are now logged in")
                return redirect(url_for("main"))

            else:
                error = "Invalid credentials, try again."

        gc.collect()

        return render_template('signin.html', error=error)

    except Exception as e:
        flash(e)
        error = "Invalid credentials, try again."
        return render_template('signin.html', error=error)
@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been logged out!")
    gc.collect()
    return redirect(url_for('main'))

"""
WTForms for handling forms
- Date Form
"""
class DateForm(Form):
    dt = DateField('Pick a Date', format="%d-%m-%Y")

""" Get All Patients """
@app.route('/get_users')
def get_users():
    # Query DB:
    cur, conn = connection()
    q_list_one = "SELECT * FROM users"
    cur.execute(q_list_one)
    users = cur.fetchall()

    users_dict = []
    for user in users:
        user_dict = {
            'Id': user[0],
            'IdUniq': user[1],
            'Name': user[2],
            'Age': user[3],
            'Date': user[4],
            'Imei': user[5],}
        users_dict.append(user_dict)

    return json.dumps(users_dict)

""" Get Patient By ID """
@app.route('/getUID', methods=["GET", "POST"])
def getUID():
    userid = request.form.get('userid')
    return userid
    # return render_template('getUID.html', userid=userid)

# @app.route('/getStartDate', methods=["GET", 'POST'])
# def getStartDate():
#     startdate = request.form.get('startdate')
#     return startdate
#
# @app.route('/getEndDate', methods=["GET", 'POST'])
# def getEndDate():
#     enddate = request.form.get('enddate')
#     return enddate
"""
Analyze Acceleration data
- Infer activity from acceleration file
"""
@app.route("/extractCSV", methods=["GET", "POST"])
def extractCSV():
    global myPath, userid, filepath, AccName
    # Declarations
    #userid = ""
    myPath = ""
    jsonFeatures = {}

    if request.method == "GET":
        #userid = request.form['userid']
        userid = request.args.get('userid')
        filepath = request.args.get('filepath')


        print (" Android must Have invoked this CSV shit")
        print("user id is : " + userid + " filepath :  "+filepath)

        # Ftp connection
        #FtpHostName = "anis.tunisia-webhosting.com"
        #FtpUser = "ahmed@anis.tunisia-webhosting.com"
        #FtpPassword = "ahmedahmed"
        #ftp = FTP(FtpHostName)
        #ftp.login(FtpUser, FtpPassword)
        path = "http://www.anis.tunisia-webhosting.com/anis.tunisia-webhosting.com/ahmed"

    ########################################
    ########################################

    # Accelerometer CSV data manipulation and features extraction

    #########################################
    #########################################
    #########################################

        def magnitude(activity):
            x2 = activity['xAxis'] * activity['xAxis']
            y2 = activity['yAxis'] * activity['yAxis']
            z2 = activity['zAxis'] * activity['zAxis']
            m2 = x2 + y2 + z2
            m = m2.apply(lambda x: math.sqrt(x))
            return m

        def windows(df, size=100):
            start = 0
            while start < df.count():
                yield start, start + size
                start += (size / 2)

        def jitter(axis, start, end):
            j = float(0)
            for i in xrange(start, min(end, axis.count())):
                if start != 0:
                    j += abs(axis[i] - axis[i - 1])
            return j / (end - start)

        def mean_crossing_rate(axis, start, end):
            cr = 0
            m = axis.mean()
            for i in xrange(start, min(end, axis.count())):
                if start != 0:
                    p = axis[i - 1] > m
                    c = axis[i] > m
                    if p != c:
                        cr += 1
            return float(cr) / (end - start - 1)

        def window_summary(axis, start, end):
            acf = stattools.acf(axis[start:end])
            acv = stattools.acovf(axis[start:end])
            sqd_error = (axis[start:end] - axis[start:end].mean()) ** 2
            return [
                jitter(axis, start, end),
                mean_crossing_rate(axis, start, end),
                axis[start:end].mean(),
                axis[start:end].std(),
                axis[start:end].var(),
                axis[start:end].min(),
                axis[start:end].max(),
                acf.mean(),  # mean auto correlation
                acf.std(),  # standard deviation auto correlation
                acv.mean(),  # mean auto covariance
                acv.std(),  # standard deviation auto covariance
                skew(axis[start:end]),
                kurtosis(axis[start:end]),
                math.sqrt(sqd_error.mean())
            ]

        def features(activity):
            for (start, end) in windows(activity['timestamp']):
                features = []
                for axis in ['xAxis', 'yAxis', 'zAxis', 'magnitude']:
                    features += window_summary(activity[axis], start, end)
                yield features

        COLUMNS = ['timestamp', 'xAxis', 'yAxis', 'zAxis']
        patho = "/var/www/html/android_login_api"

        dirPath = filepath[2:17]
        print(dirPath)
        AccName = filepath[18:39]
        print(AccName)

        #ftp.cwd("/" + dirPath)
        #print("done changing directory")

        csvfilematch = '*.csv'
        CSVPath = '/var/www/html/android_login_api/*.csv'
        cc = 0

        today = datetime.datetime.today().strftime("%d-%m-%Y %H:%M:%S")
        td = time.mktime(datetime.datetime.strptime(today, "%d-%m-%Y %H:%M:%S").timetuple())
        print(today)
        print(td)

        #for filename in ftp.nlst(csvfilematch):  # Loop - looking for WAV files
        for filename in glob.glob(CSVPath):
            cc += 1
            #print("checking FTP for the " + str(cc) + " time")
            print("checking Repository for the " + str(cc) + " time")
            print("comparing " + AccName + " and     " + filename[32:])
            print(AccName == filename[32:])

            if filename[32:] == AccName:
                #fhandle = open(filename, 'wb')
                #print('Getting ' + filename)
                #os.chdir(patho)
                #ftp.retrbinary('RETR ' + filename, fhandle.write)
                #print("stored")
                #fhandle.close()
                if os.stat("/var/www/html/android_login_api/"+AccName).st_size != 0:
                    Activity = pd.read_csv(filename, header=None, names=COLUMNS)[:3000]
                    Activity['magnitude'] = magnitude(Activity)
                    with open('/home/ahmed/flaskth/AccFtr/' + AccName + '_Features.csv', 'w') as out:
                        rows = csv.writer(out)
                        for f in features(Activity):
                            rows.writerow(f)

                    ActivityDataFeature = pd.read_csv('/home/ahmed/flaskth/AccFtr/' + AccName + '_Features.csv', header=None)
                    print(ActivityDataFeature.head())

                    test1f = numpy.loadtxt('/home/ahmed/flaskth/AccFtr/' + AccName + '_Features.csv', delimiter=",")

                    pickle_in = open('/home/ahmed/flaskth/RFCacc.pickle', 'rb')
                    c = pickle.load(pickle_in)

                    print("********************* 1 file ***********************")
                    predf1 = c.predict(test1f)
                    print("scikit learn predicted :  ")
                    print(repr(predf1))

                    myresultf1 = c.score(Xf1, yf1)
                    print
                    "score"
                    print
                    myresultf1

                    accurf1 = metrics.accuracy_score(yf1, predf1)
                    print
                    "Accuracy"
                    print
                    accurf1

                    f1scoref1 = metrics.f1_score(yf1, predf1, pos_label=list(set(yf1)))
                    print
                    "F1"
                    print
                    f1scoref1

                    cmf1 = metrics.confusion_matrix(yf1, predf1)
                    print
                    "Confusion matrix"
                    print
                    cmf1

                    Standing, Walking, Running, Stairs = 0, 0, 0, 0
                    actDict = {'Standing' : 0, 'Walking': 0, 'Running': 0, 'Stairs':0}
                    activeOrNotDict = {'Active' : 0, 'Inactive': 0}
                    for predicted_activity in numpy.nditer(predf1):
                        #print(predicted_activity)
                        if predicted_activity == 0:
                            actDict['Standing'] += 1
                            activeOrNotDict['Inactive'] += 1
                        else:
                            if predicted_activity == 1:
                                actDict['Walking'] += 1
                                activeOrNotDict['Active'] += 1
                            else:
                                if predicted_activity == 2:
                                    actDict['Running'] += 1
                                    activeOrNotDict['Active'] += 1
                                else:
                                    if predicted_activity == 3 or predicted_activity == 4:
                                        actDict['Stairs'] += 1
                                        activeOrNotDict['Active'] += 1

                    #act = max(Standing, Walking, Running, Stairs)
                    act = max(actDict.iteritems(), key=operator.itemgetter(1))[0]
                    activenot = max(activeOrNotDict.iteritems(), key=operator.itemgetter(1))[0]
                    print("The class is :" + str(act))
                    print("The user is :" + str(activenot))

                    curInsertAct, conn = connection()
                    curInsertAct.execute('''INSERT INTO activity
                                                      (uid,
                                                      act,
                                                      created_at)
                                                      VALUES (%s,
                                                              %s,
                                                              %s)''', (userid,
                                                                       str(act),
                                                                       td))
                    conn.commit()
    print("Shit from android for user id : " + str(userid) +" and the file is  :" + AccName)
    return jsonify({'USER ID': userid}, {'FILE NAME': AccName}), 201

"""
Analyze Speech daya
- Analyze phone calls made by patient o
"""
@app.route("/extractAudio", methods=["GET", "POST"])
def extractAudio():
    """ Declaration """
    global userid,starttime, endtime, myPath, numbPauses,loudness1,loudness2,dur,madmax,maxDBFS,MPA,numOfFtrs,phoneCallFtrFinal,wavo,jsonFeatures,nf
    starttime =""
    #userid = ""
    endtime = ""
    myPath = ""
    numbPauses = 0
    loudness1 = 0
    loudness2 = 0
    nf = 0
    dur = 0
    madmax = 0
    maxDBFS = 0
    MPA = 0
    numOfFtrs = 0
    wavo = ""
    jsonFeatures = {}
    phoneCallFtrFinal = pd.DataFrame()

    today = datetime.datetime.today().strftime("%d-%m-%Y %H:%M:%S")
    td = time.mktime(datetime.datetime.strptime(today, "%d-%m-%Y %H:%M:%S").timetuple())
    print(today)
    print(td)


    """ Getting parameters sent via android app"""
    if request.method == 'GET':
        print ("Android must Have invoked this")
        userid = request.args.get('userid')
        print (userid)
        audiopath = request.args.get('audiopath')
        print (audiopath)

        #userid = request.form['userid']

        # Ftp connection
        #FtpHostName = "anis.tunisia-webhosting.com"
        #FtpUser = "ahmed@anis.tunisia-webhosting.com"
        #FtpPassword = "ahmedahmed"
        #ftp = FTP(FtpHostName)
        #ftp.login(FtpUser, FtpPassword)
        #path = "http://www.anis.tunisia-webhosting.com/anis.tunisia-webhosting.com/ahmed"

        """ Phone calls processing and features extraction """

        patho = "/var/www/html/android_login_api/"
        dirPath = audiopath[2:17]
        print(dirPath)
        #audiopathwav = audiopath[:-1]
        wavName = audiopath[18:-1]
        print(wavName)

        today = datetime.datetime.today().strftime("%d-%m-%Y %H:%M:%S")
        td = time.mktime(datetime.datetime.strptime(today, "%d-%m-%Y %H:%M:%S").timetuple())
        print(today)
        print(td)

        #ftp.cwd("/"+dirPath)
        #print ("done changing directory")
        #audiofilematch = '*.wav'

        WAVPath = '/var/www/html/android_login_api/*.wav'
        cc = 0
        # for filename in ftp.nlst(csvfilematch):
        """ Loop - looking for WAV files """
        for filename in glob.glob(WAVPath):
            cc += 1
            fileWav = filename[32:]
            print ("checking Folder for the " + str(cc) + " time")
            print ("comparing " + wavName + " and     " + fileWav.replace(" ", ""))
            print (wavName == fileWav.replace(" ", ""))

            if fileWav.replace(" ", "") == wavName:
                #fhandle = open(filename, 'wb')
                #print('Getting ' + filename)
                #os.chdir("/home/ahmed/flaskth/")
                #ftp.retrbinary('RETR ' + filename, fhandle.write)
                #print("stored")
                #fhandle.close()
                print (" just do this shit")
                sound = AudioSegment.from_file(patho+fileWav)
                mypath = "/home/ahmed/flaskth/PhoneCallsFtr/"
                if not os.path.isdir(mypath):
                    os.makedirs(mypath)
                sound.export(mypath + wavName+".wav", format="wav")
                """ Split files into chunks : number of pauses """
                audio_chunks = split_on_silence(sound,
                                                # must be silent for at least 1 second
                                                min_silence_len=1000,

                                                # consider it silent if quieter than -16 dBFS
                                                silence_thresh=-16
                                                )
                for i, chunk in enumerate(audio_chunks):
                    numbPauses += i
                    out_file = "/home/ahmed/flaskth/SpeechChunks/"+wavName+"chunk{0}.wav".format(i)
                    print ("exporting", out_file)
                    chunk.export(out_file, format="wav")

                print ("numbPauses : " + str(numbPauses))
                #loudness1 = sound.rms
                #print loudness1
                #number of Frames
                nf = sound.frame_count()
                print(nf)
                # Value of loudness
                loudness2 = sound.dBFS
                print(loudness2)
                #duration
                dur = sound.duration_seconds
                #max
                madmax = sound.max
                print(madmax)
                #max possible amplitude
                MPA = sound.max_possible_amplitude
                print(MPA)
                #max dbfs
                maxDBFS = sound.max_dBFS
                print(maxDBFS)
                samplewidth = sound.sample_width

                # SELECT * FROM `phoneCallFeatures` WHERE (`created_at` BETWEEN 60 AND 1500) AND (uid = '5795028d168257.04609170')
                # SQL query to insert phone call features to DB

                #insertFtrReq1 = "INSERT INTO phoneCallFeatures (uid,npause,loudness,maxA,maxPA,created_at) VALUES (%s, %s, %s, %s, %s)", (userid,numbPauses,loudness1,madmax,MPA,wavdate)
                curInsertFtr, conn = connection()
                curInsertFtr.execute('''INSERT INTO phoneCallFeatures
                                              (uid,
                                              npause,
                                              loudness,
                                              maxA,
                                              maxPA,
                                              created_at)
                                              VALUES (%s,
                                                      %s,
                                                      %s,
                                                      %s,
                                                      %s,
                                                      %s)''',  (userid,
                                                               numbPauses,
                                                               loudness2,
                                                               madmax,
                                                               MPA,
                                                               td))
                conn.commit()

                mypath = "/home/ahmed/flaskth/PhoneCallsFtr/"
                WavToExtract = mypath+wavName+".wav"
                print ("Path of Wav to be extracted :   ======>  " + WavToExtract)
                [Fs, x] = audioBasicIO.readAudioFile(WavToExtract)

                speechPred = aT.fileClassification(WavToExtract,"/home/ahmed/RFNormDep", "randomforest")
                print("Prediction is : ");print(speechPred)

                print(aT.fileClassification(WavToExtract,"/home/ahmed/RFNormDep", "randomforest"))

                # silence removal ---- into segments
                # segments = aS.silenceRemoval(x, Fs, 0.030, 0.030, smoothWindow = 0.3, Weight = 0.6, plot=True)
                # numbPauses = len(segments) - 1

                # short term features extraction
                fw = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.1*Fs, 0.1*Fs)
                plt.subplot(2,1,1);plt.plot(fw[0,:]);plt.xlabel('Frame no');plt.ylabel('ZCR')
                plt.subplot(2,1,2);plt.plot(fw[1,:]);plt.xlabel('Frame no');plt.ylabel('Energy')
                plt.show()
                # Mid term feature extraction
                #fm = audioFeatureExtraction.mtFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs,0.050*Fs, 0.050*Fs)
                #print("MID TERM FEATURESSSSSS below")
                #print(fm)
                if not os.path.isdir(mypath):
                    os.makedirs(mypath)
                numpy.savetxt(mypath + wavName + "_AllStFtrs.csv", fw, delimiter=",")
                #numpy.savetxt(mypath+wavName+"_AllMidFtrs.csv", fm, delimiter=",")
                # read features in a data frame
                phoneCallFtrPrime = pd.read_csv(mypath + wavName+"_AllStFtrs.csv")
                numOfFtrs = len(phoneCallFtrPrime.index)
                print("number of ftrs : " + str(numOfFtrs))
                COLUMNS = []
                #for i in range(0, len(phoneCallFtrPrime.columns)):
                #    COLUMNS.append('Frame' + str(i + 1))
                #    COLUMNS.append('label')
                #phoneCallFtrFinal = pd.read_csv(mypath + filename +"_AllStFtrs.csv", header=None, names=COLUMNS)

                #numOfFtrs = len(phoneCallFtrFinal["label"])
                #phoneCallFtrFinal.label = phoneCallFtrFinal.label.astype(float).fillna(0)
                # phoneCallFtrFinal.label = phoneCallFtrFinal.label.astype(int).fillna(0)
                #for j in range(0, numOfFtrs):
                #    phoneCallFtrFinal.iloc[j]['label'] = 0

                #jsonFeatures = phoneCallFtrFinal.reset_index().to_json(orient='index')
                #jsonFeatures = phoneCallFtrFinal.to_json(orient=None)
                display(phoneCallFtrPrime.head())
                #print "JSON Features of every audio ----------------------------------------------"
                #print jsonFeatures

    print ("this script has been called and data has been handled")
    return jsonify({'USER ID': userid}, {'FILE NAME :': wavName}), 201

""" Dashboard & Front-End """
@app.route("/dashboard/<userid>", methods=["GET", "POST"])
def dashboard(userid):

    # Declarations
    global starttime, endtime
    endtime = ""
    # Ftp connection
    #FtpHostName = "anis.tunisia-webhosting.com"
    #FtpUser = "ahmed@anis.tunisia-webhosting.com"
    #FtpPassword = "ahmedahmed"
    #ftp = FTP(FtpHostName)
    #ftp.login(FtpUser, FtpPassword)
    #path = "http://www.anis.tunisia-webhosting.com/anis.tunisia-webhosting.com/ahmed"

    # forms
    form = DateForm()

    today = datetime.datetime.today().strftime("%d-%m-%Y %H:%M:%S")
    td = time.mktime(datetime.datetime.strptime(today, "%d-%m-%Y %H:%M:%S").timetuple())
    print(today)
    print(td)
    if form.validate_on_submit():
        print(" I am working with date from datepicker !!!")
        starttime = form.dt.data.strftime('%x')
        stt = time.mktime(datetime.datetime.strptime(starttime, "%d-%m-%Y").timetuple())
        print (starttime)
        print(stt)
        print(today)
        #endtime = form.dt.data.strftime('%x')


        ##########################################
        # Extract PhoneCalls features of patient depending on Date pickers values
        ##########################################

        curUser, conn = connection()
        reqUser = "SELECT * FROM users WHERE unique_id = %s"

        curUser.execute(reqUser, [userid])
        users = curUser.fetchall()

        usersDict = []
        for u in users:
            userDict = {
                'Id': u[0],
                'unique_id': u[1],
                'name': u[2],
                'age': u[3],
                'email': u[4]}
            usersDict.append(userDict)
        print(usersDict)

        ##########################################
        # Extract phone call features  depending on Date pickers values
        ##########################################

        curPhone, conn = connection()
        reqPhoneCallFtr = """SELECT *, AVG(loudness) AS 'avgLoudness', AVG(npause) AS 'avgPauses', AVG(maxPA) AS 'avgMaxPA', AVG(maxA) AS 'avgMAXA'
                                FROM phoneCallFeatures
                                WHERE  uid=%s
                                AND created_at BETWEEN  %s AND %s
                                GROUP BY created_at""", (userid, stt, td)

        curPhone.execute(*reqPhoneCallFtr)
        callsFtr = curPhone.fetchall()
        calls_ftr_dict = []
        for call in callsFtr:
            call_dict = {
                'Id': call[0],
                'IdUniq': call[1],
                'npause': call[2],
                'loudness': call[3],
                'maxA': call[4],
                'maxPA': call[5],
                'createdat': call[6],
                'avgLoudness': call[7],
                'avgPauses': int(call[8]),
                'avgMaxPA': call[9],
                'avgMAXA': call[10]}
            calls_ftr_dict.append(call_dict)

        print(calls_ftr_dict)
        for i in range(0, len(calls_ftr_dict)):
            calls_ftr_dict[i]['createdat'] = datetime.datetime.fromtimestamp(
                int(calls_ftr_dict[i]['createdat'])).strftime('%d-%m-%Y')

        print(calls_ftr_dict)

        ##########################################
        # Extract number of posts in FB  depending on Date pickers values
        ##########################################
        curFB, conn = connection()
        reqFb = """SELECT *, SUM(numpost) AS 'npost'
                       FROM fbdata
                       WHERE  uid=%s
                       AND created_at BETWEEN  %s AND %s
                        GROUP BY created_at""", (userid, stt, td)

        curFB.execute(*reqFb)
        FbNumPosts = curFB.fetchall()
        FBs_dict = []
        for post in FbNumPosts:
            FB_dict = {
                'id': post[0],
                'numpost': post[1],
                'createdat': post[2],
                'uid': post[3],
                'npost': post[4]}
            FBs_dict.append(FB_dict)

        print(FBs_dict)

        for key in range(0, len(FBs_dict)):
            FBs_dict[key]['createdat'] = datetime.datetime.fromtimestamp(float(FBs_dict[key]['createdat'])).strftime(
                '%d-%m-%Y')

        print(FBs_dict)


        return render_template('dashboardSinglePatient.html', userid=userid, starttime=starttime, form=form, usersDict=usersDict, calls_ftr_dict=calls_ftr_dict, FBs_dict=FBs_dict)

    else:
        print(" I am working within last 15 days !!!")
        last15day = datetime.datetime.today() + datetime.timedelta(-15)
        last15 = last15day.strftime("%d-%m-%Y %H:%M:%S")
        t15d = time.mktime(datetime.datetime.strptime(last15, "%d-%m-%Y %H:%M:%S").timetuple())
        t15dprime = 1472900400
        print(t15dprime)

        print("15 days ago : " + last15)
        print(t15d)
        #resultat = td - t15d
        #print("Resultat :")
        #print(resultat)

    ##########################################
    # Extract from last 15 days (default)
    ##########################################

        curUser, conn = connection()
        reqUser = "SELECT * FROM users WHERE unique_id = %s"

        curUser.execute(reqUser, [userid])
        users = curUser.fetchall()
        usersDict = []
        for u in users:
            userDict = {
                'Id': u[0],
                'unique_id': u[1],
                'name': u[2],
                'age': u[3],
                'email': u[4]}
            usersDict.append(userDict)
        print (usersDict)


        ##########################################
        # Extract phone call features from last 15 days (default)
        ##########################################

        print("############################################################")
        print("####################### phone calls ftr ###########################")
        curPhone, conn = connection()
        reqPhoneCallFtr = """SELECT *, AVG(loudness) AS 'avgLoudness', AVG(npause) AS 'avgPauses', AVG(maxPA) AS 'avgMaxPA', AVG(maxA) AS 'avgMAXA'
                            FROM phoneCallFeatures
                            WHERE  uid=%s
                            AND created_at BETWEEN  %s AND %s
                            GROUP BY created_at""", (userid, t15dprime, td)

        curPhone.execute(*reqPhoneCallFtr)
        callsFtr = curPhone.fetchall()
        print(callsFtr)

        #print("############################################")

        calls_ftr_dict = []
        for call in callsFtr:
            call_dict={
               'Id': call[0],
                'IdUniq': call[1],
                'npause': call[2],
                'loudness': call[3],
                'maxA': call[4],
                'maxPA': call[5],
                'createdat': call[6],
                'avgLoudness':call[7],
                'avgPauses': int(call[8]),
                'avgMaxPA': call[9],
                'avgMAXA': call[10]}
            calls_ftr_dict.append(call_dict)

        print (calls_ftr_dict)

        print("####################### phone calls ftr group by week ###########################")
        curPhoneWeek, conn = connection()
        reqPhoneCallFtrWeek = """SELECT FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7)) AS week_beginning, AVG(loudness) AS 'avgLoudness', AVG(npause) AS 'avgPauses', AVG(maxPA) AS 'avgMaxPA', AVG(maxA) AS 'avgMAXA' FROM `phoneCallFeatures`
                                 WHERE  uid=%s
                                GROUP BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7))
                                ORDER BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7))"""

        curPhoneWeek.execute(reqPhoneCallFtrWeek, [userid])
        callsFtrWeek = curPhoneWeek.fetchall()
        print(callsFtrWeek)

        # print("############################################")

        calls_ftr_dict_week = []
        for call in callsFtrWeek:
            call_dict = {
                'week_beginning': call[0],
                'avgLoudness': call[1],
                'avgPauses': int(call[2]),
                'avgMaxPA': call[3],
                'avgMAXA': call[4]}
            calls_ftr_dict_week.append(call_dict)

        print(calls_ftr_dict_week)

        for i in range(0, len(calls_ftr_dict)):
            calls_ftr_dict[i]['createdat'] = datetime.datetime.fromtimestamp(float(calls_ftr_dict[i]['createdat'])).strftime('%d-%m-%Y')

        print(calls_ftr_dict)

        print("################## end phone calls ftr #####################")

        ##########################################
        # Extract phone call features ALL from last 15 days (default)
        ##########################################

        cccc, conn = connection()
        req = """SELECT *
                          FROM phoneCallFeatures
                          WHERE uid=%s
                          AND created_at BETWEEN  %s AND %s
                          GROUP BY created_at""", (userid, t15dprime, td)

        cccc.execute(*req)
        Allcalls = cccc.fetchall()
        # print(calls)


        calls_ftr_dict_All = []
        for everycall in Allcalls:
            all_call_dict = {
                'Id': everycall[0],
                'IdUniq': everycall[1],
                'npause': everycall[2],
                'loudness': everycall[3],
                'maxA': everycall[4],
                'maxPA': everycall[5],
                'createdat': everycall[6],
            }
            calls_ftr_dict_All.append(all_call_dict)

        print(calls_ftr_dict_All)
        for i in range(0, len(calls_ftr_dict_All)):
            #calls_ftr_dict_All[i]['createdat'] = datetime.datetime.fromtimestamp(float(calls_ftr_dict_All[i]['createdat'])).strftime('%d-%m-%Y %H:%M:%S')
            calls_ftr_dict_All[i]['createdat'] = datetime.datetime.fromtimestamp(float(calls_ftr_dict_All[i]['createdat'])).strftime('%d-%m-%Y %H:%M:%S')

        print(calls_ftr_dict_All)




        ##########################################
        # Extract Facebook features from last 15 days (default)
        ##########################################

        curFB, conn = connection()
        reqFb = """SELECT *, SUM(numpost) AS 'npost'
                   FROM fbdata
                   WHERE  uid=%s
                   AND created_at BETWEEN  %s AND %s
                   GROUP BY created_at""", (userid, t15d, td)

        curFB.execute(*reqFb)
        FbNumPosts = curFB.fetchall()
        FBs_dict = []
        for post in FbNumPosts:
            FB_dict = {
            'id': post[0],
            'numpost': post[1],
            'createdat': post[2],
            'uid': post[3],
            'npost': post[4]}
            FBs_dict.append(FB_dict)

        print (FBs_dict)

        for key in range(0, len(FBs_dict)):
            FBs_dict[key]['createdat'] = datetime.datetime.fromtimestamp(float(FBs_dict[key]['createdat'])).strftime('%d-%m-%Y %H:%M:%S')

        print(FBs_dict)

        print("####################### by week ###############################")

        curFBweek, conn = connection()
        reqFbWeek = """SELECT FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7)) AS week_beginning, SUM(numpost) AS numberOfposts FROM fbdata
                   WHERE  uid=%s
                   GROUP BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7))
                   ORDER BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7))"""

        curFBweek.execute(reqFbWeek, [userid])
        FbNumPosts = curFBweek.fetchall()
        FBs_dict_week = []
        for post in FbNumPosts:
            FB_dict = {
                'week_beginning': post[0],
                'totalPosts': post[1],
                }
            FBs_dict_week.append(FB_dict)

        print(FBs_dict_week)


        ##########################################
        # Extract from last 15 days (default)
        ##########################################

        curBDI, conn = connection()
        reqbdi = """SELECT * FROM bdiresult
                       WHERE  uid=%s
                       AND created_at BETWEEN  %s AND %s """, (userid, t15dprime, td)

        curBDI.execute(*reqbdi)
        scoreBDI = curBDI.fetchall()
        BDIs_dict = []
        for test in scoreBDI:
            BDI_dict = {
                'id': test[0],
                'uid': test[1],
                'result': test[2],
                'score': test[3],
                'createdat': test[4]}
            BDIs_dict.append(BDI_dict)

        print(BDIs_dict)

        for key in range(0, len(BDIs_dict)):
            BDIs_dict[key]['createdat'] = datetime.datetime.fromtimestamp(float(BDIs_dict[key]['createdat'])).strftime(
                '%d-%m-%Y %H:%M:%S')

        print(BDIs_dict)

        ############################################################
        # Extract Activity of Patient (Active not active) from last 15 days
        ############################################################


        ############################################################
        # Extract Location of Patient from last 15 days
        ############################################################

        curLOC, conn = connection()
        reqloc = """SELECT * FROM locations
                           WHERE  uid=%s
                           AND created_at BETWEEN  %s AND %s """, (userid, t15d, td)

        curLOC.execute(*reqloc)
        Locations = curLOC.fetchall()
        LOCs_dict = []
        for location in Locations:
            LOC_dict = {
                'id': location[0],
                'uid': location[1],
                'lati': location[2],
                'longi': location[3],
                'place': location[4],
                'createdat': location[5]}
            LOCs_dict.append(LOC_dict)

        print(LOCs_dict)

        ############################################################
        # Extract all features ordered by week beginnings
        ############################################################

        print("####################### All features of user per week beginning ###############################")



        curAll, conn = connection()
        reqall = """SELECT users.name, users.age, SUM(fbdata.numpost) AS 'TotalNumpost', AVG(loudness) AS 'avgLoudness', AVG(npause) AS 'avgPauses', AVG(maxPA) AS 'avgMaxPA', AVG(maxA) AS 'avgMAXA' , FROM_DAYS(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -2, 7)) AS week_beginning
                    FROM `phoneCallFeatures`
                    INNER JOIN fbdata
                    ON phoneCallFeatures.uid=fbdata.uid
                    INNER JOIN users
                    ON phoneCallFeatures.uid=users.unique_id
                    WHERE  phoneCallFeatures.uid=%s
                    GROUP BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -2, 7))
                    ORDER BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -2, 7))"""

        reqall2="""SELECT PatientName, PatientAge ,TotalNumpost , avgLoudness, avgPauses, avgMAXA, week_beginning, week_beginning + INTERVAL 14 DAY AS weekend

            FROM (
            SELECT users.name AS 'PatientName', users.age AS 'PatientAge', (SELECT SUM(numpost) FROM fbdata WHERE fbdata.uid=phoneCallFeatures.uid ) AS 'TotalNumpost', AVG(loudness) AS 'avgLoudness', AVG(npause) AS 'avgPauses', AVG(maxA) AS 'avgMAXA' , FROM_DAYS(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -2, 14)) AS 'week_beginning'
            FROM `phoneCallFeatures`
            INNER JOIN users
	        ON phoneCallFeatures.uid=users.unique_id
            GROUP BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -2, 14))
            ) AS summary
            WHERE phoneCallFeatures.uid=%s
            ORDER BY week_beginning"""

        curAll.execute(reqall, [userid])
        dataperweek = curAll.fetchall()
        datas_dict = []
        for data in dataperweek:
            data_dict = {
                'name': data[0],
                'age': data[1],
                'totalPosts': data[2],
                'avgLoudness': data[3],
                'avgPauses': data[4],
                'avgMaxPA': data[5],
                'avgMaxA': data[6],
                'weekbeginning': data[7]}
            datas_dict.append(data_dict)

        print("""
        ********************************************
        This is the query to output it all in weeks
        ********************************************
        """)
        print(datas_dict)


        return render_template('dashboardSinglePatient.html', userid=userid, last15=last15, form=form, usersDict=usersDict, calls_ftr_dict=calls_ftr_dict,calls_ftr_dict_All=calls_ftr_dict_All, calls_ftr_dict_week=calls_ftr_dict_week, FBs_dict=FBs_dict, FBs_dict_week=FBs_dict_week, BDIs_dict=BDIs_dict, LOCs_dict=LOCs_dict)

@app.route("/predictions/<userid>", methods=["GET", "POST"])
def predictions(userid):

    # Declarations
    global starttime, endtime
    endtime = ""
    # Ftp connection
    #FtpHostName = "anis.tunisia-webhosting.com"
    #FtpUser = "ahmed@anis.tunisia-webhosting.com"
    #FtpPassword = "ahmedahmed"
    #ftp = FTP(FtpHostName)
    #ftp.login(FtpUser, FtpPassword)
    #path = "http://www.anis.tunisia-webhosting.com/anis.tunisia-webhosting.com/ahmed"

    # forms
    form = DateForm()

    today = datetime.datetime.today().strftime("%d-%m-%Y %H:%M:%S")
    td = time.mktime(datetime.datetime.strptime(today, "%d-%m-%Y %H:%M:%S").timetuple())
    print(today)
    print(td)
    if form.validate_on_submit():
        print(" I am working with date from datepicker !!!")
        starttime = form.dt.data.strftime('%x')
        stt = time.mktime(datetime.datetime.strptime(starttime, "%d-%m-%Y").timetuple())
        print (starttime)
        print(stt)
        print(today)
        #endtime = form.dt.data.strftime('%x')


        ##########################################
        # Extract PhoneCalls features of patient depending on Date pickers values
        ##########################################

        curUser, conn = connection()
        reqUser = "SELECT * FROM users WHERE unique_id = %s"

        curUser.execute(reqUser, [userid])
        users = curUser.fetchall()

        usersDict = []
        for u in users:
            userDict = {
                'Id': u[0],
                'unique_id': u[1],
                'name': u[2],
                'age': u[3],
                'email': u[4]}
            usersDict.append(userDict)
        print(usersDict)

        ##########################################
        # Extract phone call features  depending on Date pickers values
        ##########################################

        curPhone, conn = connection()
        reqPhoneCallFtr = """SELECT *, AVG(loudness) AS 'avgLoudness', AVG(npause) AS 'avgPauses', AVG(maxPA) AS 'avgMaxPA', AVG(maxA) AS 'avgMAXA'
                                FROM phoneCallFeatures
                                WHERE  uid=%s
                                AND created_at BETWEEN  %s AND %s
                                GROUP BY created_at""", (userid, stt, td)

        curPhone.execute(*reqPhoneCallFtr)
        callsFtr = curPhone.fetchall()
        calls_ftr_dict = []
        for call in callsFtr:
            call_dict = {
                'Id': call[0],
                'IdUniq': call[1],
                'npause': call[2],
                'loudness': call[3],
                'maxA': call[4],
                'maxPA': call[5],
                'createdat': call[6],
                'avgLoudness': call[7],
                'avgPauses': int(call[8]),
                'avgMaxPA': call[9],
                'avgMAXA': call[10]}
            calls_ftr_dict.append(call_dict)

        print(calls_ftr_dict)
        for i in range(0, len(calls_ftr_dict)):
            calls_ftr_dict[i]['createdat'] = datetime.datetime.fromtimestamp(
                int(calls_ftr_dict[i]['createdat'])).strftime('%d-%m-%Y')

        print(calls_ftr_dict)

        ##########################################
        # Extract number of posts in FB  depending on Date pickers values
        ##########################################
        curFB, conn = connection()
        reqFb = """SELECT *, SUM(numpost) AS 'npost'
                       FROM fbdata
                       WHERE  uid=%s
                       AND created_at BETWEEN  %s AND %s
                        GROUP BY created_at""", (userid, stt, td)

        curFB.execute(*reqFb)
        FbNumPosts = curFB.fetchall()
        FBs_dict = []
        for post in FbNumPosts:
            FB_dict = {
                'id': post[0],
                'numpost': post[1],
                'createdat': post[2],
                'uid': post[3],
                'npost': post[4]}
            FBs_dict.append(FB_dict)

        print(FBs_dict)

        for key in range(0, len(FBs_dict)):
            FBs_dict[key]['createdat'] = datetime.datetime.fromtimestamp(float(FBs_dict[key]['createdat'])).strftime(
                '%d-%m-%Y')

        print(FBs_dict)


        return render_template('dashboardSinglePatient.html', userid=userid, starttime=starttime, form=form, usersDict=usersDict, calls_ftr_dict=calls_ftr_dict, FBs_dict=FBs_dict)

    else:
        print(" I am working within last 15 days !!!")
        last15day = datetime.datetime.today() + datetime.timedelta(-15)
        last15 = last15day.strftime("%d-%m-%Y %H:%M:%S")
        t15d = time.mktime(datetime.datetime.strptime(last15, "%d-%m-%Y %H:%M:%S").timetuple())
        t15dprime = 1472900400
        print(t15dprime)

        print("15 days ago : " + last15)
        print(t15d)
        #resultat = td - t15d
        #print("Resultat :")
        #print(resultat)

    ##########################################
    # Extract from last 15 days (default)
    ##########################################

        curUser, conn = connection()
        reqUser = "SELECT * FROM users WHERE unique_id = %s"

        curUser.execute(reqUser, [userid])
        users = curUser.fetchall()
        usersDict = []
        for u in users:
            userDict = {
                'Id': u[0],
                'unique_id': u[1],
                'name': u[2],
                'age': u[3],
                'email': u[4]}
            usersDict.append(userDict)
        print (usersDict)


        ##########################################
        # Extract phone call features from last 15 days (default)
        ##########################################

        print("############################################################")
        print("####################### phone calls ftr ###########################")
        curPhone, conn = connection()
        reqPhoneCallFtr = """SELECT *, AVG(loudness) AS 'avgLoudness', AVG(npause) AS 'avgPauses', AVG(maxPA) AS 'avgMaxPA', AVG(maxA) AS 'avgMAXA'
                            FROM phoneCallFeatures
                            WHERE  uid=%s
                            AND created_at BETWEEN  %s AND %s
                            GROUP BY created_at""", (userid, t15dprime, td)

        curPhone.execute(*reqPhoneCallFtr)
        callsFtr = curPhone.fetchall()
        print(callsFtr)

        #print("############################################")

        calls_ftr_dict = []
        for call in callsFtr:
            call_dict={
               'Id': call[0],
                'IdUniq': call[1],
                'npause': call[2],
                'loudness': call[3],
                'maxA': call[4],
                'maxPA': call[5],
                'createdat': call[6],
                'avgLoudness':call[7],
                'avgPauses': int(call[8]),
                'avgMaxPA': call[9],
                'avgMAXA': call[10]}
            calls_ftr_dict.append(call_dict)

        print (calls_ftr_dict)

        print("####################### phone calls ftr group by week ###########################")
        curPhoneWeek, conn = connection()
        reqPhoneCallFtrWeek = """SELECT FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7)) AS week_beginning, AVG(loudness) AS 'avgLoudness', AVG(npause) AS 'avgPauses', AVG(maxPA) AS 'avgMaxPA', AVG(maxA) AS 'avgMAXA' FROM `phoneCallFeatures`
                                 WHERE  uid=%s
                                GROUP BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7))
                                ORDER BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7))"""

        curPhoneWeek.execute(reqPhoneCallFtrWeek, [userid])
        callsFtrWeek = curPhoneWeek.fetchall()
        print(callsFtrWeek)

        # print("############################################")

        calls_ftr_dict_week = []
        for call in callsFtrWeek:
            call_dict = {
                'week_beginning': call[0],
                'avgLoudness': call[1],
                'avgPauses': int(call[2]),
                'avgMaxPA': call[3],
                'avgMAXA': call[4]}
            calls_ftr_dict_week.append(call_dict)

        print(calls_ftr_dict_week)

        for i in range(0, len(calls_ftr_dict)):
            calls_ftr_dict[i]['createdat'] = datetime.datetime.fromtimestamp(float(calls_ftr_dict[i]['createdat'])).strftime('%d-%m-%Y')

        print(calls_ftr_dict)

        print("################## end phone calls ftr #####################")

        ##########################################
        # Extract phone call features ALL from last 15 days (default)
        ##########################################

        cccc, conn = connection()
        req = """SELECT *
                          FROM phoneCallFeatures
                          WHERE uid=%s
                          AND created_at BETWEEN  %s AND %s
                          GROUP BY created_at""", (userid, t15dprime, td)

        cccc.execute(*req)
        Allcalls = cccc.fetchall()
        # print(calls)


        calls_ftr_dict_All = []
        for everycall in Allcalls:
            all_call_dict = {
                'Id': everycall[0],
                'IdUniq': everycall[1],
                'npause': everycall[2],
                'loudness': everycall[3],
                'maxA': everycall[4],
                'maxPA': everycall[5],
                'createdat': everycall[6],
            }
            calls_ftr_dict_All.append(all_call_dict)

        print(calls_ftr_dict_All)
        for i in range(0, len(calls_ftr_dict_All)):
            #calls_ftr_dict_All[i]['createdat'] = datetime.datetime.fromtimestamp(float(calls_ftr_dict_All[i]['createdat'])).strftime('%d-%m-%Y %H:%M:%S')
            calls_ftr_dict_All[i]['createdat'] = datetime.datetime.fromtimestamp(float(calls_ftr_dict_All[i]['createdat'])).strftime('%d-%m-%Y %H:%M:%S')

        print(calls_ftr_dict_All)




        ##########################################
        # Extract Facebook features from last 15 days (default)
        ##########################################

        curFB, conn = connection()
        reqFb = """SELECT *, SUM(numpost) AS 'npost'
                   FROM fbdata
                   WHERE  uid=%s
                   AND created_at BETWEEN  %s AND %s
                   GROUP BY created_at""", (userid, t15d, td)

        curFB.execute(*reqFb)
        FbNumPosts = curFB.fetchall()
        FBs_dict = []
        for post in FbNumPosts:
            FB_dict = {
            'id': post[0],
            'numpost': post[1],
            'createdat': post[2],
            'uid': post[3],
            'npost': post[4]}
            FBs_dict.append(FB_dict)

        print (FBs_dict)

        for key in range(0, len(FBs_dict)):
            FBs_dict[key]['createdat'] = datetime.datetime.fromtimestamp(float(FBs_dict[key]['createdat'])).strftime('%d-%m-%Y %H:%M:%S')

        print(FBs_dict)

        print("####################### by week ###############################")

        curFBweek, conn = connection()
        reqFbWeek = """SELECT FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7)) AS week_beginning, SUM(numpost) AS numberOfposts FROM fbdata
                   WHERE  uid=%s
                   GROUP BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7))
                   ORDER BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(created_at)) -2, 7))"""

        curFBweek.execute(reqFbWeek, [userid])
        FbNumPosts = curFBweek.fetchall()
        FBs_dict_week = []
        for post in FbNumPosts:
            FB_dict = {
                'week_beginning': post[0],
                'totalPosts': post[1],
                }
            FBs_dict_week.append(FB_dict)

        print(FBs_dict_week)


        ##########################################
        # Extract from last 15 days (default)
        ##########################################

        curBDI, conn = connection()
        reqbdi = """SELECT * FROM bdiresult
                       WHERE  uid=%s
                       AND created_at BETWEEN  %s AND %s """, (userid, t15dprime, td)

        curBDI.execute(*reqbdi)
        scoreBDI = curBDI.fetchall()
        BDIs_dict = []
        for test in scoreBDI:
            BDI_dict = {
                'id': test[0],
                'uid': test[1],
                'result': test[2],
                'score': test[3],
                'createdat': test[4]}
            BDIs_dict.append(BDI_dict)

        print(BDIs_dict)

        for key in range(0, len(BDIs_dict)):
            BDIs_dict[key]['createdat'] = datetime.datetime.fromtimestamp(float(BDIs_dict[key]['createdat'])).strftime(
                '%d-%m-%Y %H:%M:%S')

        print(BDIs_dict)

        ############################################################
        # Extract Activity of Patient (Active not active) from last 15 days
        ############################################################


        ############################################################
        # Extract Location of Patient from last 15 days
        ############################################################

        curLOC, conn = connection()
        reqloc = """SELECT * FROM locations
                           WHERE  uid=%s
                           AND created_at BETWEEN  %s AND %s """, (userid, t15d, td)

        curLOC.execute(*reqloc)
        Locations = curLOC.fetchall()
        LOCs_dict = []
        for location in Locations:
            LOC_dict = {
                'id': location[0],
                'uid': location[1],
                'lati': location[2],
                'longi': location[3],
                'place': location[4],
                'createdat': location[5]}
            LOCs_dict.append(LOC_dict)

        print(LOCs_dict)

        ############################################################
        # Extract all features ordered by week beginnings
        ############################################################

        print("####################### All features of user per week beginning ###############################")



        curAll, conn = connection()
        reqall = """SELECT users.name, users.age, SUM(fbdata.numpost) AS 'TotalNumpost', AVG(loudness) AS 'avgLoudness', AVG(npause) AS 'avgPauses', AVG(maxPA) AS 'avgMaxPA', AVG(maxA) AS 'avgMAXA' , FROM_DAYS(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -2, 7)) AS week_beginning
                    FROM `phoneCallFeatures`
                    INNER JOIN fbdata
                    ON phoneCallFeatures.uid=fbdata.uid
                    INNER JOIN users
                    ON phoneCallFeatures.uid=users.unique_id
                    WHERE  phoneCallFeatures.uid=%s
                    GROUP BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -2, 7))
                    ORDER BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -2, 7))"""

        reqall2="""SELECT PatientName, PatientAge ,TotalNumpost , avgLoudness, avgPauses, avgMAXA, week_beginning, week_beginning + INTERVAL 14 DAY AS weekend

            FROM (
            SELECT users.name AS 'PatientName', users.age AS 'PatientAge', (SELECT SUM(numpost) FROM fbdata WHERE fbdata.uid=phoneCallFeatures.uid ) AS 'TotalNumpost', AVG(loudness) AS 'avgLoudness', AVG(npause) AS 'avgPauses', AVG(maxA) AS 'avgMAXA' , FROM_DAYS(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -2, 14)) AS 'week_beginning'
            FROM `phoneCallFeatures`
            INNER JOIN users
	        ON phoneCallFeatures.uid=users.unique_id
            GROUP BY FROM_DAYS(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -MOD(TO_DAYS(FROM_UNIXTIME(phoneCallFeatures.created_at)) -2, 14))
            ) AS summary
            WHERE phoneCallFeatures.uid=%s
            ORDER BY week_beginning"""

        curAll.execute(reqall, [userid])
        dataperweek = curAll.fetchall()
        datas_dict = []
        for data in dataperweek:
            data_dict = {
                'name': data[0],
                'age': data[1],
                'totalPosts': data[2],
                'avgLoudness': data[3],
                'avgPauses': data[4],
                'avgMaxPA': data[5],
                'avgMaxA': data[6],
                'weekbeginning': data[7]}
            datas_dict.append(data_dict)

        print(datas_dict)
        return render_template('predictions.html', userid=userid, last15=last15, form=form, usersDict=usersDict, calls_ftr_dict=calls_ftr_dict,calls_ftr_dict_All=calls_ftr_dict_All, calls_ftr_dict_week=calls_ftr_dict_week, FBs_dict=FBs_dict, FBs_dict_week=FBs_dict_week, BDIs_dict=BDIs_dict, LOCs_dict=LOCs_dict)

@app.route("/dashboardall", methods=["GET", "POST"])
def dashboardall(userid):
    return render_template('dashboardAll.html')

""" Outlier detection function"""
def mad_based_outlier(points, thresh=3.5):
    """
       Returns a boolean array with True if points are outliers and False
       otherwise.
       Parameters:
       -----------
           points : An numobservations by numdimensions array of observations
           thresh : The modified z-score to use as a threshold. Observations with
               a modified z-score (based on the median absolute deviation) greater
               than this value will be classified as outliers.
       Returns:
       --------
           mask : A numobservations-length boolean array.
       References:
       ----------
           Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
           Handle Outliers", The ASQC Basic References in Quality Control:
           Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
       """
    if len(points.shape) == 1:
        points = points[:,None]
    median = numpy.median(points, axis=0)
    diff = numpy.sum((points - median)**2, axis=-1)
    diff = numpy.sqrt(diff)
    med_abs_deviation = numpy.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9999)
