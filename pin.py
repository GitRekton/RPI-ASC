import pigpio
import argparse

speed = 0

parser = argparse.ArgumentParser()
parser.add_argument("pin", help="a pinnumber",type=int)
parser.add_argument("value", help="logical state: 1 or 0",type=int)

args = parser.parse_args()

pi = pigpio.pi()
pi.set_PWM_dutycycle(args.pin, args.value)

