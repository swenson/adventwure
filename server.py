from __future__ import print_function
# Copyright (c) 2016-2021 Twilio Inc.

import os
from Queue import Queue
import threading
import time

import psycopg2
dburl = os.getenv('DATABASE_URL')
if not dburl:
    dburl = 'dbname=test user=cswenson'
conn = psycopg2.connect(dburl)
cur = conn.cursor()
try:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS adventure (
            num VARCHAR(32) PRIMARY KEY,
            state BYTEA,
            created TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            modified TIMESTAMP WITH TIME ZONE DEFAULT NOW());
    """)
    conn.commit()
except:
    pass
cur.close()

from twilio import twiml
from flask import Flask, request, jsonify, Request
from werkzeug.datastructures import ImmutableOrderedMultiDict
class MyRequest(Request):
    """Request subclass to override request parameter storage"""
    parameter_storage_class = ImmutableOrderedMultiDict
class MyFlask(Flask):
    """Flask subclass using the custom request class"""
    request_class = MyRequest

app = MyFlask(__name__)




from interpret import Game

class TwilioHandler(object):
    def __init__(self):
        self.outqueue = Queue()
        self.inqueue = Queue()
    def read(self):
        return self.inqueue.get()
    def write(self, data):
        self.outqueue.put(data)

sid = os.getenv('TWILIO_SID')
token = os.getenv('TWILIO_TOKEN')
from_num = os.getenv('TWILIO_NUMBER')

states = {}

def run_for(from_, inp):
    try:
        cur = conn.cursor()
        inp = str(inp).upper().strip()
        inp = inp[:20] # commands shouldn't be longer than this

        cur.execute("SELECT state FROM adventure WHERE num = %s", (from_,))
        row = cur.fetchone()
        exists = row is not None
        ignore_input = False
        new_game = False

        if inp == 'RESET' or inp == 'QUIT' or inp == 'PURGE':
            if from_ in states:
                del states[from_]
                exists = False # force a reset
                cur.execute("DELETE FROM adventure WHERE num = %s", (from_,))
        if inp == 'PURGE':
            resp = twiml.Response()
            text = 'Your data has been purged from the database. Text back to start a new game in the future if you like.'
            resp.message(text)
            return str(resp)

        if not exists:
            print('starting new game for', from_)
            handler = TwilioHandler()
            game = Game(handler)
            t = threading.Thread(target=game.go)
            t.daemon = True
            t.start()
            states[from_] = [handler, game, t]
            ignore_input = True
            new_game = True

        if exists and from_ not in states:
            # load from backup
            handler = TwilioHandler()
            game = Game(handler)
            t = threading.Thread(target=game.go)
            t.daemon = True
            t.start()
            states[from_] = [handler, game, t]
            # wait fot it to boot
            while not game.waiting():
                time.sleep(0.001)
            # empty the queues
            while not handler.outqueue.empty():
                handler.outqueue.get_nowait()
            game.setstate(row[0])
            states[from_] = [handler, game, t]

        handler, game, _ = states[from_]
        if not ignore_input:
            handler.inqueue.put(inp)
        time.sleep(0.001)
        while not game.waiting():
            time.sleep(0.001)
        text = ''
        while not text:
            while not handler.outqueue.empty():
                text += handler.outqueue.get()
                time.sleep(0.001)

        # now save the game state to the database
        state = game.getstate()
        if exists:
            cur.execute("UPDATE adventure SET state = %s, modified = NOW() WHERE num = %s", (psycopg2.Binary(state), from_))
        else:
            cur.execute("INSERT INTO adventure (num, state) VALUES (%s,%s)", (from_, psycopg2.Binary(state)))
        conn.commit()

        if new_game:
            text = 'Welcome to Adventure! Type RESET or QUIT to restart the game. Type PURGE to be removed from our database.\n\n' + text
        return text
    finally:
        cur.close()

@app.route("/incoming-voice", methods=['GET', 'POST'])
def voice_reply():
    print('Form', request.form)
    from_ = request.form['DialogueSid'][2:34]
    inp = ''
    if 'Field_word1_Value' in request.form:
      inp += ' ' + request.form.getlist('Field_word1_Value')[-1]
    if 'Field_word2_Value' in request.form and len((request.values.get('CurrentInput') or '').split(' ')) > 1:
            inp += ' ' + request.form.getlist('Field_word2_Value')[-1]
    inp = inp.strip()[:20]
    if inp == '':
        inp = request.values.get('CurrentInput') or ''
    inp = inp.strip().upper()
    inp = inp.replace('.', '')
    inp = inp.replace(',', '')
    inp = str(inp)
    print('Recognized input %s' % inp)

    text = run_for(from_, inp)
    print('Output %s' % text)
    actions = []
    if inp:
        text = 'I heard ' + inp + '. ' + text
    actions.append({'say': {'speech': text}})
    actions.append({'listen': True})
    resp = {'actions': actions}
    return jsonify(resp)

@app.route("/incoming-sms", methods=['GET', 'POST'])
def sms_reply():
    from_ = str(request.values.get('From'))
    inp = str(request.values.get('Body', ''))
    text = run_for(from_, inp)
    resp = twiml.Response()
    resp.message(text)
    return str(resp)

@app.route('/')
def hello_world():
    return 'Hello, World!'
