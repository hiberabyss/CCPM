#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Hongbo Liu <hbliu@freewheel.tv>
# Date: 2017-07-29
# Last Modified Date: 2017-07-29
# Last Modified By: Hongbo Liu <hbliu@freewheel.tv>

from math import *
import random
import sys

class Task():
    def __init__(self,
                 ID,
                 pre_ids,
                 a,
                 b,
                 c,
                 r11,
                 r12,
                 r13,
                 r21,
                 r22,
                 r23,
                 r31,
                 r32,
                 r33,
                 sigma):

        self.ID = ID
        self.pre_ids = list(pre_ids)
        self.a = int(a)
        self.b = int(b)
        self.c = int(c)
        self.sigma = float(sigma)

        self.k = 0.5

        self.e = -1
        self.es = -1
        self.ef = -1
        self.ls = -1
        self.lf = -1

        self.ff = -1

        self.r = [[int(r11), int(r12), int(r13)],
                  [int(r21), int(r22), int(r23)],
                  [int(r31), int(r32), int(r33)]]
        #  self.r = [float(r1), float(r2), float(r3), float(r4), float(r5)]
        self.r_total = [48, 53, 42]

    def e_f1(self, k):
        return self.a + sqrt(k * (self.c - self.a) * (self.b - self.a))

    def e_f2(self, k):
        return (k * (self.c - self.b + self.d - self.a) + self.a + self.b) / 2.0

    def e_f3(self, k):
        return self.c - sqrt((self.c - self.a) * (1-k) * (self.c - self.b))

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
        return self.gete(0.9, 0) - self.gete(0.5, 0)

    def gete(self, k=0.5, mode=1):
        import numpy
        #  return numpy.random.normal(self.b, self.sigma)
        base = self.a
        #  random_num = random.randint(0,2)
        if self.ID in list('ABCDUT'):
          base = self.a
        if self.ID in list('EHIJKLNRS'):
          base = self.b
        if self.ID in list('MOPQFG'):
          base = self.c

        return numpy.random.normal(base, self.sigma)

        if mode > 0:
            return [self.a, self.b, self.c][self.get_node_type()]

        e1 = self.e_f1(k)
        e3 = self.e_f3(k)

        if self.a < e1 and e1 <= self.b:
            return e1
        elif self.b <= e3 and e3 < self.c:
            return e3

        return -1

    def get_node_type(self):
        if self.ID in list('ABCDUT'):
            return 0

        if self.ID in list('FGMOPQ'):
            return 2

        return 1

    def get_rt(self):
        max_rt = 0
        for i in range(len(self.r_total)):
            max_rt = max(max_rt, self.r[self.get_node_type()][i] * 1.0 / self.r_total[i])
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


def calc_pb(critical_chain, task_map):
    sum_pb = 0

    for name in critical_chain:
        t = task_map[name]
        sum_pb += t.per_task_pb()

    return sqrt(sum_pb) * get_network_complexity([critical_chain], task_map)


def get_e_sum(critical_chain, task_map):
    sum = 0
    for item in critical_chain:
        sum += task_map[item].gete(0.5)
    return sum


def calc_es_ef(tasks, tmap, t):
    if t.es >= 0:
        return

    if len(t.pre_ids) == 0:
        t.es = 0
        t.ef = t.gete(t.k)
        start_node.append(t.ID)
        #  print "Start Node: " + t.ID
        return

    for item in t.pre_ids:
        pre_t = tmap[item]
        calc_es_ef(tasks, tmap, pre_t)
        t.es = max(t.es, pre_t.ef)

    t.ef = t.es + t.gete(t.k)


def calc_ls_lf(tasks, tmap, t, k, lf):
    if t.lf >= 0:
        return

    t.lf = lf
    t.ls = t.lf - t.gete(k)

    for item in t.pre_ids:
        pre_t = tmap[item]
        calc_ls_lf(tasks, tmap, pre_t, k, t.ls)


def get_task_with_max_ef(tasks):
    max_t = None
    max_ef = -1

    for t in tasks:
        if t.lf >= 0:
            continue
        if t.ef > max_ef:
            max_ef = t.ef
            max_t = t

    return max_t


def read_tasks(filename):
    taskf = open(filename)
    lines = taskf.readlines()
    taskf.close()
    tasks = []
    task_map = {}

    for line in lines:
        items = line.split(',')
        t = Task(items[0],
                 items[1],
                 items[2],
                 items[3],
                 items[4],
                 items[6],
                 items[7],
                 items[8],
                 items[9],
                 items[10],
                 items[11],
                 items[12],
                 items[13],
                 items[14],
                 items[15])
        tasks.append(t)
        task_map[items[0]] = t

    return tasks, task_map


def print_tasks(tasks):
    print "node,es,ef,ls,lf,af,rt,TF,FF, diff"
    for t in tasks:
        print "%s,%s,%s,%s,%s,%s,%s,%s,%s,%f" % (t.ID, t.es, t.ef, t.ls, t.lf, t.get_af(), t.get_rt(), (t.ls - t.es), t.ff, t.get_diff())


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
            if len(last_node.pre_ids) == 0:
                continue

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


def get_network_complexity(paths, task_map):
    path = []
    for p in paths:
        if len(p) > len(path):
            path = p

    pre_ids_sum = 0
    for n in path:
        pre_ids_sum += len(task_map[n].pre_ids)

    return pre_ids_sum * 1.0 / (len(path))


def calc_all_fb(tasks, task_map, cc, end_node):
    nc_end_nodes = get_non_critial_end_node(tasks, task_map, cc, end_node)

    for node_id in nc_end_nodes:
        ncc_paths = get_non_critial_chain(tasks, task_map, node_id)

        max_fb = max(map(lambda x: calc_pb(x, task_map), ncc_paths))

        node = task_map[node_id]
        print "%s FB:FF:Final %s %s %s %s" % (ncc_paths, max_fb, node.ff, min(max_fb, node.ff), get_network_complexity(ncc_paths, task_map))


def random_lists(cnt):
    res = []
    for j in range(cnt):
        l = []
        for i in range(19):
            l.append(random.random())
        res.append(l)

    return res


def get_nodes_max_ef(tasks):
    max_ef = 0
    for t in tasks:
        calc_es_ef(tasks, task_map, t)
        if t.ef > max_ef:
            max_ef_id = t.ID
        max_ef = max(max_ef, t.ef)

    return max_ef


def distribute(ef_list, cnt):
    total = 0.0
    res_map = {}

    for r in ef_list:
        total += r
        level = int(r/100)
        if level in res_map.keys():
            res_map[level] += 1
        else:
            res_map[level] = 1

    print "Average Max EF: %f" % (total/cnt)
    print res_map


def get_average_max_ef(tasks):
    cnt = 1000

    rlists = random_lists(cnt)

    res = []

    for i in range(cnt):
        tmp_tasks = list(tasks)
        for j in range(19):
            tmp_tasks[j].k = rlists[i][j]
            tmp_tasks[j].es = -1
            tmp_tasks[j].ef = -1
        res.append(get_nodes_max_ef(tmp_tasks))

    distribute(res, cnt)
    for i in range(0, 1000, 100):
        distribute(res[i:i+100], 100)


def get_critical_chain(k=0.5):
    tasks, task_map = read_tasks("./tasks2.csv")
    max_ef = 0
    max_ef_id = ''

    for t in tasks:
        calc_es_ef(tasks, task_map, t)
        if t.ef > max_ef:
            max_ef_id = t.ID
        max_ef = max(max_ef, t.ef)

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

    return cc, task_map

def cc_stats():
    task_map = {}
    cc_array = []

    counts = 10000
    for i in range(counts):
        cc, task_map = get_critical_chain()
        cc_array.append(cc)

    import collections
    task_statics = collections.defaultdict(int)
    time_cnt = collections.defaultdict(int)
    for cc in cc_array:
        for t in cc:
            task_statics[t] += 1
        time_cnt[floor(get_e_sum(cc, task_map))] += 1
        print("Critical Tasks: %s \t Total E: %f" % (cc, get_e_sum(cc, task_map)))

    import matplotlib.pyplot as plt
    lists = sorted(time_cnt.items())

    sum = 0
    sum_350 = 0
    cdf = [[], []]
    for item in lists:
      if item[0] <= 350:
        sum_350 += item[1]
      sum += item[1]
      cdf[0].append(item[0])
      cdf[1].append(sum * 1.0 / counts)
    #  print lists
    #  x, y = zip(*lists)
    val_350 = sum_350*1.0/counts
    plt.annotate("[350," + str(val_350) + "]", (350, val_350), (350, val_350))
    plt.plot(cdf[0], cdf[1])
    plt.plot(350, sum_350*1.0/counts, 'ks')
    plt.show()

    print task_statics


if __name__ == "__main__":
    cc_stats()
    #  cc = get_critical_chain(task_map)
    #  print cc

    #  print "Max ef: %s" % max_ef
    #  while True:
        #  t = get_task_with_max_ef(tasks)
        #  if t is None:
            #  break
        #  print "End Node: %s %s" % (t.ID, t.ef)
        #  end_node.append(t.ID)
        #  calc_ls_lf(tasks, task_map, t, k, max_ef)

    #  for t in tasks:
        #  t.get_free_float(tasks, task_map, max_ef)

    #  print_tasks(tasks, task_map, max_ef)

    #  print get_e_sum(cc, task_map)
    #  print calc_pb(cc, task_map)

    #  calc_all_fb(tasks, task_map, cc, end_node)

    #  cc_tasks = map(lambda x: task_map[x], cc)

# vim:expandtab:
