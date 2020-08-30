import numpy as np
import pygame as pg
import sys
import csv
from sklearn.neural_network import MLPClassifier as nn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pickle
import neat

stepSize = 32
x_steps = 15
y_steps = 20
WIDTH = x_steps * stepSize
HEIGHT = y_steps * stepSize
generation = 0

road1_Y = 0
road2_Y = -HEIGHT
road1 = pg.transform.scale(pg.image.load('Car_images/road.jpg'), [WIDTH, HEIGHT])
road2 = pg.transform.scale(pg.image.load('Car_images/road.jpg'), [WIDTH, HEIGHT])

hole1 = pg.transform.scale(pg.image.load('Car_images/strainer.png'), (64, 64))
hole2 = pg.transform.scale(pg.image.load('Car_images/strainer.png'), (64, 64))
blast = pg.transform.scale(pg.image.load('Car_images/explosion.png'), (128, 128))

x_step_half = x_steps - (x_steps - int(x_steps / 2))
y_step_half = y_steps - (y_steps - int(y_steps / 2))

hole1_x = stepSize * np.random.randint(0, x_step_half)
hole1_y = -stepSize

hole2_x = stepSize * np.random.randint(x_step_half, x_steps)
hole2_y = 0

hole1_rect = None
hole2_rect = None

screen = pg.display.set_mode((WIDTH, HEIGHT))


class Car:

    boundary = 35

    def __init__(self):
        pg.font.init()
        self.text = pg.font.Font('Font/HoltwoodOneSC-Regular.ttf', 16)
        self.car = pg.transform.scale(pg.transform.rotate(pg.image.load('Car_images/car.png'), 180), (96, 96))
        self.X = stepSize * x_step_half
        self.Y = stepSize * (y_step_half + 4)
        self.sensor1 = None
        self.sensor2 = None
        self.sensor3 = None
        self.sensor_dist1 = 10
        self.sensor_dist2 = 10
        self.sensor_dist3 = 10
        self.dir = 0
        self.car_rect = None
        self.dist = 0.0

    def update_car(self, dir, i):

        if dir == 1:
            self.X = self.X + stepSize
            if self.X >= WIDTH - 96:
                self.X = self.X - stepSize
            screen.blit(self.car, [self.X, self.Y])
        if dir == -1:
            self.X = self.X - stepSize
            if self.X <= 0:
                self.X = self.X + stepSize
            screen.blit(self.car, [self.X, self.Y])

        self.car_rect = pg.Rect((self.X + 15, self.Y), (64, 96))
        #pg.draw.rect(screen, pg.Color('red'), self.car_rect)
        screen.blit(self.car, [self.X, self.Y])
        dist1 = self.update_sensor()
        self.dist = self.dist + 0.1
        self.score_text(np.round(self.dist, 1), i)
        return dist1

    def update_sensor(self):
        sensor_length = 352
        self.sensor1 = pg.Rect([self.X + 8, self.Y - sensor_length + 48], [80, sensor_length])
        #pg.draw.rect(screen, pg.Color('red'), self.sensor1, 1)


        wall_l = pg.Rect([16, 0], [32, HEIGHT])
        #pg.draw.rect(screen, pg.Color('red'), wall_l, 1)

        wall_r = pg.Rect([WIDTH-48, 0], [32, HEIGHT])
        #pg.draw.rect(screen, pg.Color('red'), wall_r, 1)

        Y1 = self.sensor1.midbottom[1]/stepSize

        y1 = hole1_y/stepSize
        y2 = hole2_y / stepSize

        if self.sensor1.colliderect(hole1_rect):
            self.sensor_dist1 = np.sqrt((Y1 - y1)**2)-2
        elif self.sensor1.colliderect(hole2_rect):
            self.sensor_dist1 = np.sqrt((Y1 - y2)**2)-2
        else:
            self.sensor_dist1 = sensor_length/stepSize
        #
        # if self.sensor2.colliderect(wall_l):
        #     self.sensor_dist2 = np.sqrt(((self.sensor2[0] + sensor_length/3 + 32 - wall_l[0])/stepSize)**2)
        # else:
        #     self.sensor_dist2 = (sensor_length/stepSize)/3
        #
        # if self.sensor3.colliderect(wall_r):
        #     self.sensor_dist3 = np.sqrt(((self.sensor3[0] - wall_r[0])/stepSize)**2)
        # else:
        #     self.sensor_dist3 = (sensor_length/stepSize)/3

        return np.round(self.sensor_dist1, 2)

    def collision_check(self):

        if self.car_rect.colliderect(hole1_rect) or self.car_rect.colliderect(hole2_rect):
            screen.blit(blast, [self.X+15, self.Y])
            return False
        else:
            return True

    def score_text(self, x, i):
        i = i*20
        score = self.text.render(f'Distance: {x} m', False, pg.Color('white'))
        screen.blit(score, [10, i + 10])
    

def update_road():
    global road1_Y, road2_Y, screen
    road1_Y += stepSize
    screen.blit(road1, [0, road1_Y])
    if road1_Y >= HEIGHT:
        road1_Y = - HEIGHT

    road2_Y += stepSize
    screen.blit(road2, [0, road2_Y])
    if road2_Y >= HEIGHT:
        road2_Y = - HEIGHT
        

def update_hole():
    global hole1_x, hole1_y, hole2_x, hole2_y, x_step_half, hole1_rect, hole2_rect
    hole1_y += stepSize
    hole1_rect = pg.Rect([hole1_x, hole1_y], [32, 32])
    screen.blit(hole1, [hole1_x, hole1_y])
    if hole1_y >= HEIGHT:
        hole1_x = stepSize*np.random.randint(1, x_step_half)
        hole1_y = stepSize*np.random.randint(-15, -10)

    hole2_y += stepSize
    hole2_rect = pg.Rect([hole2_x, hole2_y], [32, 32])
    screen.blit(hole2, [hole2_x, hole2_y])
    if hole2_y >= HEIGHT:
        hole2_x = stepSize*np.random.randint(x_step_half, x_steps-2)
        hole2_y = 0


def car_game(genomes, config):
    clock = pg.time.Clock()
    # Init NEAT
    nets = []
    cars = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        # Init my cars
        cars.append(Car())

    pg.init()
    generation_font = pg.font.Font('Font/HoltwoodOneSC-Regular.ttf', 20)
    font = pg.font.Font('Font/HoltwoodOneSC-Regular.ttf', 14)

    # Main loop
    global generation
    generation = generation + 1
    remain_cars = len(genomes)
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit(0)

        # Input my data and get result from network
        for j, car in enumerate(cars):
            output = nets[j].activate([car.sensor_dist1])
            i = output.index(max(output))
            if i == 0:
                car.dir = -1
            elif i == 1:
                car.dir = 0
            else:
                car.dir = 1

        # Update car and fitness
        update_road()
        update_hole()
        for i, car in enumerate(cars):
            d1 = car.update_car(car.dir, i)
            if car.collision_check():
                genomes[i][1].fitness += 0.1
                if d1 < 8:
                    genomes[i][1].fitness -= 0.1
                if d1 > 8:
                    genomes[i][1].fitness += 0.1
            else:
                genomes[i][1].fitness -= 1
                nets.pop(i)
                cars.pop(i)
                remain_cars -= 1


        # check
        if remain_cars == 0:
            break

        text = generation_font.render("Generation : " + str(generation), True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (WIDTH / 2, HEIGHT - 80)
        screen.blit(text, text_rect)

        text = font.render("remain cars : " + str(remain_cars), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (WIDTH / 2, HEIGHT - 50)
        screen.blit(text, text_rect)

        pg.display.update()
        pg.display.set_caption("Car Game [NEAT]")

        clock.tick(0)


if __name__ == "__main__":
    # Set configuration file
    config_path = "config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    p.run(car_game, 1000)
