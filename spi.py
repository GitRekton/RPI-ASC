import spidev
import time
import os
import pigpio

CS_pin = 8
DELAY = 0.5
pi = pigpio.pi()

spi = spidev.SpiDev()
spi.open(0,0)

print "Reset complete"

for i in range(1000):
        spi.xfer([80])
        time.sleep(0.001)
        ok = spi.readbytes(5)
        print ok
	time.sleep(0.01)
	
