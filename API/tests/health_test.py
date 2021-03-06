# psutil document : https://psutil.readthedocs.io/en/latest/
# 참고코드(system monitoring) : https://github.com/camilochs/Chasar/blob/master/src/core/clientnode/__init__.py
# 참고코드(scheduler) : https://medium.com/@kevin.michael.horan/scheduling-tasks-in-django-with-the-advanced-python-scheduler-663f17e868e6

import json
import psutil
import datetime
import platform
import netifaces
from time import gmtime, strftime
from apscheduler.schedulers.background import BackgroundScheduler
import os

class HealthTest :
    healthInfo={}

    def __init__(self):
        self.boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
        self.cores = psutil.cpu_count()
        self. os, name, version, _, _, _ = platform.uname()
        self.version = version.split('-')[0]
        print("HealthTest 객체가 생성됨")

    def getHealthInfo(self):
        return self.healthInfo

    def updateHeatlhInfo(self):
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        pidcount = len(psutil.pids())
        swap_info = psutil.swap_memory()
        virtual_info = psutil.virtual_memory()

        self.healthInfo = json.dumps({
            "computer_utc_clock": str(datetime.datetime.utcnow()),
            "computer_clock": str(datetime.datetime.now()),
            "hostname": platform.node(),
            "mac_address": self.mac_address(),
            "ipv4_interfaces": self.internet_addresses(),
            "os": {
                "name": self.os,
                "version": self.version
            },
            "cpu": {
                "percent_used": cpu_percent,
                "core_num": self.cores
            },
            "virtual_memory": {
                "total_bytes": virtual_info.total,
                "total_bytes_used": virtual_info.used,
                "percent_used": virtual_info.percent
            },
            "ram_memory": {
                "total_bytes": memory_info.total,
                "total_bytes_used": memory_info.used,
                "percent_used": memory_info.percent
            },
            "swap_memory": {
                "total_bytes": swap_info.total,
                "total_bytes_used": swap_info.used,
                "percent_used": swap_info.percent
            },
            "disk": {
                "total_bytes": disk_info.total,
                "total_bytes_used": disk_info.used,
                "total_bytes_free": disk_info.free,
                "percent_used": disk_info.percent
            },
            "running_since": self.boot_time.strftime("%A %d. %B %Y"),
            "pidcount": pidcount,
            "update_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }).encode()
        print(self.healthInfo)

    def mac_address(self):
        """
        Return the mac address of computer.
        """
        interface = netifaces.ifaddresses('en0')
        info = interface[netifaces.AF_LINK]
        if info:
            return interface[netifaces.AF_LINK][0]["addr"]


    def internet_addresses(self):
        """
        Return the info network.
        """
        interface = netifaces.ifaddresses('en0')
        info = interface[netifaces.AF_INET]
        if info:
            return interface[netifaces.AF_INET]


    def pids_active(self, pids_computer):
        """
        This function find pids of computer and return the valid.
        """
        pid_valid = {}
        for pid in pids_computer:
            data = None
            try:
                process = psutil.Process(pid)
                data = {"pid": process.pid,
                        "status": process.status(),
                        "percent_cpu_used": process.cpu_percent(interval=0.0),
                        "percent_memory_used": process.memory_percent()}

            except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
                data = None

            if data is not None:
                pid_valid[process.name()] = data
        return pid_valid


    def find_procs_by_name(self, name):
        "Return a list of processes matching 'name'."
        ls = []
        names=name.split(" ")
        nameSize=len(names)
        counter=0;
        for p in psutil.process_iter(attrs=["cmdline", "name","exe"]):
            cmdline=p.info['cmdline']

            if cmdline is not None :
                for cmd in cmdline:
                    if cmd in names:
                        counter=counter+1

                if(counter == nameSize):
                    ls.append(p)
                counter=0

        return ls

if __name__ == "__main__":

    health_info= HealthTest()
    # health_info.updateHeatlhInfo()
    # print(health_info.getHealthInfo())

    if health_info.find_procs_by_name('rabbit'):
        print('Yes a rabbitmq process was running')
    else:
        print('No rabbitmq process was running')

    if health_info.find_procs_by_name('smartcity beat'):
        print('Yes a beat process was running')
    else:
        print('No beat process was running')

    if health_info.find_procs_by_name('smartcity worker'):
        print('Yes a worker process was running')
    else:
        print('No worker process was running')