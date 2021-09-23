# Adventure for Twilio!

By [Christopher Swenson](mailto:cswenson@twilio.com), Twilio Inc.

![Adventure Graphic](https://upload.wikimedia.org/wikipedia/commons/2/25/ADVENT_--_Crowther_Woods.png)

[Adventure](https://en.wikipedia.org/wiki/Colossal_Cave_Adventure)
was the first text adventure and the first interactive fiction game,
and one of the first viral games.

## How to Play

Just text anything to [+1 (669) 238-3683](tel:+16692383683) ("669 ADVENT3") to start a game.
Send `RESET` to restart the game.
Case doesn't matter when sending commands.

## Technical Notes

There are four primary components to this software:

* [advdat.77-03-31.txt](advdat.77-03-31.txt) – The 1977 Adventure data file
* [advf4.77-03-31.txt](advf4.77-03-31.txt) – The 1977 Adventure source code (in FORTRAN IV)
* [interpret.py](interpret.py) – A FORTRAN IV interpreter, written in Python 2.7, that supports the PDP-10 architecture.
* [server.py](server.py) – A web server that connects the FOTRAN interpreter to the Twilio API and to a database to store user data.

Essentially, we run the FORTRAN source code through our interpreter, and when
the code requests input from the teletype, we instead feed it the
last command sent via Twilio SMS. If there is no command buffered up, the interpreter
sleeps, and waits to be woken up when there is a command. Any teletype output is
sent to the user via Twilio SMS.

The entire state of the FOTRAN interpreter is pickled (serialized), compressed,
and sent to the database after every command is processed.
This is about 18 KB per player.

## History and Challenges

The original source code was written in (the now obsolete) FORTRAN IV,
which is no longer supported by any modern compilers.
In addition, it was written for the [PDP-10](https://en.wikipedia.org/wiki/PDP-10)
(released in 1966), whose FORTRAN IV variant had a lot of quirks, which make
using the old source code challenging.

Some of the fun quirks:

* The PDP-10 is a 36-bit machine.
* It supports a very old version of ASCII, possibly 1963 or 1965.
* It supports character types by packing 5 7-bit ASCII characters into a 36-bit word. The least significant bit is left as a zero, and they are packed "left-to-right" inside the word.
* It is meant to read from disk, tape drive, or teletype, and write to a teletype. So we have to provide emulation for these.

## License

MIT License

Copyright (c) 2016–2021 Twilio Inc.

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
