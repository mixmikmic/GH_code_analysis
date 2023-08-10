# run this code first, only need to run it once
import RPi.GPIO as GPIO
import mcpi.minecraft as minecraft
import mcpi.block as block
from picamera import PiCamera
from time import sleep
import datetime

mc = minecraft.Minecraft.create()

def teleport():
    mc.player.setPos(0, 100, 0)

def flash(code, t):
    GPIO.output(code, True)
    sleep(t)
    GPIO.output(code, False)
    sleep(t)

def bulldoze():
    x, y, z, = mc.player.getTilePos()
    mc.setBlocks(x-10, 0, z-10, x+10, 20, z+10, block.AIR)

def photobooth(LED):
    x, y, z, = mc.player.getTilePos()
    mc.setBlocks(x+3, y, z+3, x+5, y+2, z+5, block.GLOWING_OBSIDIAN)
    mc.setBlocks(x+3, y, z+4, x+4, y+1, z+4, block.AIR)
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED, GPIO.OUT)
    
    camera = PiCamera()

    try:
        while True:
            pos = mc.player.getTilePos()
            if (pos.x, pos.y, pos.z) == (x+4, y, z+4):
                mc.postToChat("You are in the Photobooth!")
                sleep(1)
                mc.postToChat("Smile!")
                flash(LED, 0.5) #1s
                camera.start_preview()
                flash(LED, 0.5) #1s
                flash(LED, 0.5) #1s
                camera.capture('/home/pi/Desktop/image%s.jpg' % datetime.datetime.now().isoformat())
                camera.stop_preview()
            sleep(3)
    finally:
        camera.stop_preview()
        camera.close()
        GPIO.cleanup()

def lava():
    x, y, z, = mc.player.getTilePos()
    mc.setBlocks(x+4, 4, z+4, x+5, 5, z+5, block.LAVA.id)

def water():
    x, y, z, = mc.player.getTilePos()
    mc.setBlocks(x+4, 4, z+4, x+5, 5, z+5, block.WATER.id)

def tnt():
    x, y, z, = mc.player.getTilePos()
    mc.setBlocks(x+4, 0, z+4, x+5, 1, z+5, block.TNT.id, 1)

def tower():
    x, y, z, = mc.player.getTilePos()
    mc.setBlocks(x+3, 0, z+3, x+3, 200, z+3, block.MELON.id)

def do_with_countdown(BUTTON, LED_LIST, action):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    for LED in LED_LIST:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED, GPIO.OUT)

    try:
        for LED in LED_LIST:
            GPIO.output(LED, True)

        while GPIO.input(BUTTON):
            sleep(0.1)

        print("button pressed")

        for LED in LED_LIST:
            sleep(1)
            GPIO.output(LED, False)
        sleep(1)
        action();

    finally:
        GPIO.cleanup()

def do_on_press(BUTTON, action):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    try:
        while GPIO.input(BUTTON):
            sleep(0.1)
        print("button pressed")
        action();

    finally:
        GPIO.cleanup()

teleport()

bulldoze()

photobooth(17)

do_with_countdown(4, [17], water)

do_on_press(4, bulldoze)



