#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Hongbo Liu <hbliu@freewheel.tv>
# Date: 2017-07-29
# Last Modified Date: 2017-07-29
# Last Modified By: Hongbo Liu <hbliu@freewheel.tv>

from math import *

class Task():
    def __init__(self, ID, pre_ids, a, b, c, d, r1, r2, r3, r4, r5):
        self.ID = ID
        self.pre_ids = list(pre_ids)
        self.a = int(a)
        self.b = int(b)
        self.c = int(c)
        self.d = int(d)
        self.e = -1
        self.es = -1
        self.ef = -1
        self.ls = -1
        self.lf = -1

        self.ff = -1

        self.r = [float(r1), float(r2), float(r3), float(r4), float(r5)]
        self.r_total = [8, 6, 5, 20, 7]

        self.r1 = -1
        self.r2 = -1
        self.r3 = -1
        self.r4 = -1
        self.r5 = -1

    def e_f1(self, k):
        return self.a + sqrt(k * (self.c - self.b + self.d - self.a) * (self.b-self.a))

    def e_f2(self, k):
        return (k * (self.c - self.b + self.d - self.a) + self.a + self.b) / 2.0

    def e_f3(self, k):
        return self.d - sqrt((self.c - self.b + self.d - self.a) * (1-k) * (self.d - self.c))

    def get_af(self):
        tf = self.ls - self.es
        if 0 <= tf and tf <= 5:
            return 1.1
        elif 5 < tf and tf <= 10:
            return 1
        elif 10 < tf:
            return 0.9
        return 0

    def get_diff(self):
        return self.gete(0.9) - self.gete(0.5)

    def gete(self, k):
        e1 = self.e_f1(k)
        e2 = self.e_f2(k)
        e3 = self.e_f3(k)

        if self.a < e1 and e1 <= self.b:
            return e1
        elif self.b < e2 and e2 < self.c:
            return e2
        elif self.c <= e3 and e3 < self.d:
            return e3

        return -1

    def get_rt(self):
        max_rt = 0
        for i in range(5):
            max_rt = max(max_rt, self.r[i] * 1.0 / self.r_total[i])
        return max_rt

    def per_task_pb(self):
        return ((1 + self.get_rt()) * self.get_af() * self.get_diff()) ** 2

    def per_task_fb(self):
        return 1

    def get_children(self, tasks):
        children = []
        for node in tasks:
            if self.ID in node.pre_ids:
                children.append(node.ID)

        return children

    def get_free_float(self, tasks, task_map, max_ef):
        children = self.get_children(tasks)

        min_children_es = max_ef
        for nodeID in children:
            node = task_map[nodeID]
            min_children_es = min(min_children_es, node.es)

        self.ff = min_children_es - self.ef
        return min_children_es - self.ef

start_node = []
end_node = []

def calc_pb(critical_chain):
    sum_pb = 0
    for t in critical_chain:
        sum_pb += t.per_task_pb()
    return sqrt(sum_pb)

def calc_es_ef(tasks, tmap, t, k):
    if t.es >= 0: return

    if len(t.pre_ids) == 0:
        t.es = 0
        t.ef = t.gete(k)
        start_node.append(t.ID)
        print "Start Node: " + t.ID
        return

    for item in t.pre_ids:
        pre_t = tmap[item]
        calc_es_ef(tasks, tmap, pre_t, k)
        t.es = max(t.es, pre_t.ef)

    t.ef = t.es + t.gete(k)

def calc_ls_lf(tasks, tmap, t, k, lf):
    if t.lf >= 0: return

    t.lf = lf
    t.ls = t.lf - t.gete(k)

    for item in t.pre_ids:
        pre_t = tmap[item]
        calc_ls_lf(tasks, tmap, pre_t, k, t.ls)
        #  t.es = max(t.es, pre_t.ef)

def get_task_with_max_ef(tasks):
    max_t = None
    max_ef = -1

    for t in tasks:
        if t.lf >= 0: continue
        #  if t.lf < 0 and abs(t.ef - MAX_EF) <= 0.00001: return t
        if t.ef > max_ef:
            max_ef = t.ef
            max_t = t

    return max_t

task_map = {}

def read_tasks(filename):
    taskf = open(filename)
    lines = taskf.readlines()
    taskf.close()
    tasks = []

    for line in lines:
        items = line.split(',')
        t = Task(items[0], items[1], items[2], items[3], items[4], items[5], items[7], items[8], items[9], items[10], items[11])
        tasks.append(t)
        task_map[items[0]] = t

    return tasks

def print_tasks(tasks, task_map, max_ef):
    print "node,es,ef,ls,lf,af,rt,TF,FF"
    for t in tasks:
        print "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (t.ID, t.es, t.ef, t.ls, t.lf, t.get_af(), t.get_rt(), t.get_diff(), (t.ls - t.es), t.ff)

def get_critical_chain(tasks, task_map, max_ef_id):
    cc = [max_ef_id]
    node = task_map[max_ef_id]
    while len(node.pre_ids) > 0:
        max_ef = 0
        max_id = ''
        for node_id in node.pre_ids:
            if task_map[node_id].ef >= max_ef:
                max_ef = task_map[node_id].ef
                max_id = node_id
        node = task_map[max_id]
        cc.append(max_id)
    # for t in tasks:
    #     if t.es == t.ls:
    #         cc.append(t.ID)
    return cc

def calc_fb_via_map(chain_map, task_map):
    tasks = []
    for item in chain_map:
        tasks.append(task_map[item])
    return calc_pb(tasks)

def get_non_critial_end_node(tasks, task_map, critical_chain, end_node):
    nc_end_nodes = []

    for node_id in end_node:
        if node_id not in critical_chain:
            nc_end_nodes.append(node_id)

    for node_id in critical_chain:
        node = task_map[node_id]
        for pre_node in node.pre_ids:
            if pre_node in critical_chain: continue
            nc_end_nodes.append(pre_node)

    return nc_end_nodes

def get_non_critial_chain(tasks, task_map, node_id):
    paths = [[node_id]]

    while True:
        is_all_start_node = True
        for p in paths:
            if len(task_map[p[-1]].pre_ids) > 0:
                is_all_start_node = False
                break

        if is_all_start_node: break

        for p in paths:
            last_node = task_map[p[-1]]
            if len(last_node.pre_ids) == 0: continue

            min_ff = min(map(lambda x: task_map[x].ff, last_node.pre_ids))

            paths.remove(p)
            for pre_id in last_node.pre_ids:
                pre_node = task_map[pre_id]
                pre_ff = pre_node.ff
                if pre_ff > min_ff: continue

                tmp_path = list(p)
                tmp_path.append(pre_id)
                paths.append(tmp_path)

    return paths

def calc_all_fb(tasks, task_map, cc, end_node):
    nc_end_nodes = get_non_critial_end_node(tasks, task_map, cc, end_node)

    for node_id in nc_end_nodes:
        ncc_paths = get_non_critial_chain(tasks, task_map, node_id)

        max_fb = max(map(lambda x: calc_fb_via_map(x, task_map), ncc_paths))

        node = task_map[node_id]
        print "%s FB:FF:Final %s %s %s" % (ncc_paths, max_fb, node.ff, min(max_fb, node.ff))

if __name__ == "__main__":
    tasks = read_tasks("./tasks.csv")
    max_ef = 0
    max_ef_id = ''

    k = 0.5
    for t in tasks:
        calc_es_ef(tasks, task_map, t, k)
        if t.ef > max_ef:
            max_ef_id = t.ID
        max_ef = max(max_ef, t.ef)

    print "Max ef: %s" % max_ef
    while True:
        t = get_task_with_max_ef(tasks)
        if t is None: break
        print "End Node: %s %s" % (t.ID, t.ef)
        end_node.append(t.ID)
        calc_ls_lf(tasks, task_map, t, k, max_ef)

    for t in tasks:
        t.get_free_float(tasks, task_map, max_ef)

    cc = get_critical_chain(tasks, task_map, max_ef_id)

    print cc

    print_tasks(tasks, task_map, max_ef)

    calc_all_fb(tasks, task_map, cc, end_node)

    cc_tasks = map(lambda x: task_map[x], cc)

    print calc_pb(cc_tasks)

# vim:expandtab:
