from k8sclient.keywords import (
    list_ready_nodes,
    wait_for_pod_state,
    SUCCEEDED,
    tail_pod_logs,
    delete_pod,
    NOT_FOUND
)
from k8sclient.Components import (
    PodBuilder,
    HostPathVolume,
    RBDVolume,
    CephFSVolume,
    EmptyDirVolume
)

image = "127.0.0.1:30100/library/alpine-fio:v1"
args = "--output-format=json"
namespace = "monkey"
nodes = list_ready_nodes()
FIO_DIR = "/mnt/fio"
ceph_monitors = "10.19.137.144:6789,10.19.137.145:6789,10.19.137.146:6789"
ceph_pool = "monkey"
ceph_fstype = "xfs"
ceph_secret = "ceph-secret"  # need create beforehand
# must match the regex [a-z0-9]([-a-z0-9]*[a-z0-9])?
empty_dir = EmptyDirVolume("empty-dir-fio", FIO_DIR)
ceph_fs = CephFSVolume(
            "cephfs",
            FIO_DIR,
            monitors=ceph_monitors,
            secret_name=ceph_secret,
            fs_path="/",
            sub_path="monkey-fio"
        )
rbd = RBDVolume(
        "rbd",
        FIO_DIR,
        fs_type=ceph_fstype,
        image="default",
        pool=ceph_pool,
        monitors=ceph_monitors,
        secret_name=ceph_secret,
        sub_path="fio",
    )

volumes = {
    "empty_dir": empty_dir,
    "rbd": rbd,
    "ceph_fs": ceph_fs
}
io_engines = ["libaio", "mmap", "posixaio", "sync"]


def test(node):
    print node
    pod_name = "fio-" + "-".join(node.split("."))
    reports = []
    for n, v in volumes.items():
        print n
        for e in io_engines:
            print e
            PodBuilder(
                pod_name,
                namespace,
            ).set_node(
                node
            ).add_container(
                pod_name + "-container",
                image=image,
                args=args,
                limits={'cpu': '1', 'memory': '512Mi'},
                requests={'cpu': '0', 'memory': '0'},
                volumes=[v],
                FIO_DIR=FIO_DIR,
                IOENGINE=e
            ).deploy()
            # wait to complete
            wait_for_pod_state(namespace, pod_name, timeout=3600, expect_status=SUCCEEDED)
            logs = tail_pod_logs(namespace, pod_name).strip()
            # delete the pod
            delete_pod(namespace, pod_name)
            wait_for_pod_state(namespace, pod_name, timeout=240, expect_status=NOT_FOUND)
            # report = json.loads(logs)
            report = eval(logs)
            print report
            for job in report["jobs"]:
                print "READ:", job['read']['bw'], "KB/s"
                print "WRITE:", job['write']['bw'], "KB/s"
            reports.append({
                "vtype": n,
                "io_engine": e,
                "read(MB/s)": float(report["jobs"][0]['read']['bw'])/1024,
                "write(MB/S)": float(report["jobs"][0]['write']['bw'])/1024
            })
    return reports

disk_io_512M = test("10.19.137.150")

import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

df2 = pd.DataFrame(disk_io_512M)
df2.set_index(["vtype", "io_engine"])

df.set_index(["vtype", "io_engine"]) #140 8G file 1G ram

paas_ext4 = pd.DataFrame([
    {"vtype": "rbd/ext4", "io_engine": "libaio", "read": "43.3MB/s", "write": "43.4MB/s"},
    {"vtype": "rbd/ext4", "io_engine": "mmap", "read": "45.4MB/s", "write": "45.5MB/s"},
    {"vtype": "rbd/ext4", "io_engine": "posixaio", "read": "4465kB/s", "write": "4466kB/s"},
    {"vtype": "rbd/ext4", "io_engine": "sync", "read": "48.8MB/s", "write": "48.8MB/s"},
])
# paas_ext4.set_index(["vtype", "io_engine"])

paas_xfs = pd.DataFrame([
    {"vtype": "rbd/xfs", "io_engine": "libaio", "read": "40.5MB/s", "write": "40.5MB/s"},
    {"vtype": "rbd/xfs", "io_engine": "mmap", "read": "43.3MB/s", "write": "43.3MB/s"},
    {"vtype": "rbd/xfs", "io_engine": "posixaio", "read": "4381kB/s", "write": "4383kB/s"},
    {"vtype": "rbd/xfs", "io_engine": "sync", "read": "46.2MB/s", "write": "46.2MB/s"},
])
paas_xfs.set_index(["vtype", "io_engine"])

