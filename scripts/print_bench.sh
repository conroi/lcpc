#!/bin/sh
# (C) 2021 lcpc authors

set -u

CTN="$1".txt
PVS="$1"_pvs.txt

grep '^[0-9][0-9]:' $CTN | awk '{print($2/1e9);}' > ~/tmp/foo_ctime
grep '^[0-9][0-9]:' $PVS | awk '{print($2/1e9);}' > ~/tmp/foo_ptime
grep '^[0-9][0-9]:' $PVS | awk '{print($3/1e9);}' > ~/tmp/foo_vtime
grep '^[0-9][0-9]:' $PVS | awk '{print($4/1024);}' > ~/tmp/foo_comm
