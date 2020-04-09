# Priority dictionary using binary heaps

from __future__ import generators


class PriorityDictionary(dict):
    def __init__(self):
        self.__heap = list()
        dict.__init__(self)

    def smallest(self):
        """
        Find smallest item after removing deleted items from heap.
        :return:
        """
        if not len(self):
            pass
        heap = self.__heap

        while heap[0][1] not in self or self[heap[0][1]] != heap[0][0]:
            last_item = heap.pop()
            insertion_point = 0

            while True:
                small_child = insertion_point*2 + 1
                if small_child + 1 < len(heap) and \
                        heap[small_child] > heap[small_child + 1]:
                    small_child += 1
                if small_child >= len(heap) or last_item <= heap[small_child]:
                    heap[insertion_point] = last_item
                    break
                heap[insertion_point] = heap[small_child]
                insertion_point = small_child
            return heap[0][1]

    def __iter__(self):
        """
        Create destructive sorted iterator of PriorityDictionary.
        :return:
        """
        def iterfn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]

        return iterfn()

    def __setitem__(self, key, value):
        """
        Change value sorted in dictionary and add corresponding pair to heap.
        Rebuild the heap if the number of deleted items grows too large, to
        avoid memory linkage.
        :param key:
        :param value:
        :return:
        """
        dict.__setitem__(self, key, value)
        heap = self.__heap
        if len(heap) > 2 * len(self):
            self.__heap = [(v, k) for k, v in self.items()]
            self.__heap.sort()  # builtin sort likely faster than O(n) heapify
        else:
            new_pair = (value, key)
            insertion_point = len(heap)
            heap.append(None)
            while insertion_point > 0 and \
                    new_pair < heap[(insertion_point - 1) // 2]:
                heap[insertion_point] = heap[(insertion_point - 1) // 2]
                insertion_point = (insertion_point - 1) // 2
            heap[insertion_point] = new_pair

    def setdefault(self, key, value):
        """
        Reimplement setdefault to call customized __setitem__.
        :param key:
        :param value:
        :return:
        """
        if key not in self:
            self[key] = value

        return self[key]
