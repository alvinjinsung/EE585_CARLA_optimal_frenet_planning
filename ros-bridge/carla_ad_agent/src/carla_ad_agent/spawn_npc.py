#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import logging
import random

argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    '--host',
    metavar='H',
    default='127.0.0.1',
    help='IP of the host server (default: 127.0.0.1)')
argparser.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
argparser.add_argument(
    '-n', '--number-of-vehicles',
    metavar='N',
    default=10,
    type=int,
    help='number of vehicles (default: 10)')
argparser.add_argument(
    '-w', '--number-of-walkers',
    metavar='W',
    default=50,
    type=int,
    help='number of walkers (default: 50)')
argparser.add_argument(
    '--safe',
    action='store_true',
    help='avoid spawning vehicles prone to accidents')
argparser.add_argument(
    '--filterv',
    metavar='PATTERN',
    default='vehicle.*',
    help='vehicles filter (default: "vehicle.*")')
argparser.add_argument(
    '--filterw',
    metavar='PATTERN',
    default='walker.pedestrian.*',
    help='pedestrians filter (default: "walker.pedestrian.*")')
argparser.add_argument(
    '-tm_p', '--tm_port',
    metavar='P',
    default=8000,
    type=int,
    help='port to communicate with TM (default: 8000)')
argparser.add_argument(
    '--sync',
    action='store_true',
    help='Synchronous mode execution')
args = argparser.parse_args()


vehicles_list = []
client = carla.Client(args.host, args.port)
client.set_timeout(10.0)

def main():

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    walkers_list = []
    all_id = []

    try:

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        world = client.get_world()

        # # # Destroy all npcs
        # actor_list = world.get_actors()
        # actor = actor_list.find(85)
        # print("Currently actor list: {}").format(len(actor_list))
        # if actor is not None:
        #     actor.destory()

        synchronous_master = False

        if args.sync:
            settings = world.get_settings()
            print("setting: {}").format(settings.synchronous_mode)
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        blueprints = world.get_blueprint_library().filter(args.filterv)
        blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

        filtered = []
        for x in blueprints:
            if x.id.endswith('a2'):
                if int(x.get_attribute('number_of_wheels')) == 4:
                    print(x)
                    filtered.append(x)       
        # blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        # blueprints = [x for x in blueprints if x.get_attribute['id'].endswith('audi.a2')]
        print("blueprints size: {}").format(len(filtered))
        # spawn_points = world.get_map().get_spawn_points()

        spawn_points = []
        
        # obstacle #1 
        ob1 = carla.Transform()
        ob1.location.x = -485.49
        ob1.location.y = 281.83
        ob1.location.z = 0.0
        ob1.rotation.roll = 0.0
        ob1.rotation.pitch = 0.0
        ob1.rotation.yaw = 250
        spawn_points.append(ob1)

        # obstacle #2
        ob2 = carla.Transform()
        ob2.location.x = -491.83
        ob2.location.y = 269.24
        ob2.location.z = 0.0
        ob2.rotation.roll = 0.0
        ob2.rotation.pitch = 0.0
        ob2.rotation.yaw = 260
        spawn_points.append(ob2)

        # dynamic obs #1
        ob3 = carla.Transform()
        ob3.location.x = -337.83
        ob3.location.y = 414.05
        ob3.location.z = 0.0
        ob3.rotation.roll = 0.0
        ob3.rotation.pitch = 0.0
        ob3.rotation.yaw = 210
        spawn_points.append(ob3)
        
        number_of_spawn_points = len(spawn_points)

        # if args.number_of_vehicles < number_of_spawn_points:
        #     random.shuffle(spawn_points)
        # elif args.number_of_vehicles > number_of_spawn_points:
        #     msg = 'requested %d vehicles, but could only find %d spawn points'
        #     logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
        #     args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(filtered)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform))
            # batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        print("Batch size {}").format(len(batch))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
                print(response.actor_id)
        
        traffic_manager.global_percentage_speed_difference(30.0)
        
        while True:
            if args.sync and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    finally:

        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        time.sleep(0.5)
        
    finally:
        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        time.sleep(0.5)
        print('\ndone.')

