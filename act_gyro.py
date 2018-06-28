import pigpio
import argparse

speed = 0

parser = argparse.ArgumentParser()
parser.add_argument("state", help="adjust the logic level",type=int)
parser.add_argument("pin", help="pin selection",type=int)

args = parser.parse_args()

pi = pigpio.pi()
#pi.set_PWM_dutycycle(18, args.speed)
if args.state is 1 or args.state is 0:
    pi.write(args.pin, args.state)
