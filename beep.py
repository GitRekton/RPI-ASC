import pigpio
import argparse

speed = 0

parser = argparse.ArgumentParser()
parser.add_argument("speed", help="adjust the car speed",type=int)

args = parser.parse_args()

pi = pigpio.pi()
pi.set_PWM_dutycycle(18, args.speed)

