import numpy as np
import copy
import png


class MemoryBlock:
    def __init__(self, creation=None, last_use=None, size=0, copy_block=None):

        if copy_block is not None:
            self.creation = copy_block.creation
            self.last_use = copy_block.last_use

            self.allocation_order = copy_block.allocation_order

            self.mem_offset = copy_block.mem_offset
        else:
            self.creation = creation
            self.last_use = last_use
            self.size = size

            self.allocation_order = None
            self.mem_offset = None

    def overlaps(self, adjacent):

        if isinstance(adjacent, MemoryBlock):
            no_overlap = self.creation > adjacent.last_use or self.last_use < adjacent.creation
            return not no_overlap
        elif isinstance(adjacent, int):
            return adjacent >= self.creation and adjacent <= self.last_use
        else:
            print("Error: Non MemoryBlock or int types passed to MemoryBlock.overlaps!")
            return false

    def allocated(self):
        return self.mem_offset is not None


class MemoryRegion:
    def __init__(self, start=0, end=None):
        self.start = start
        self.end = end

    def get_carve_result(self, new_region):
        """
        returns a list of memory regions left over after this region has had the new region removed from it
        can return an empty list if the new region completely overlaps this one, a single region if it is
        clipped or two regions if this region is bisected
        :param new_region:
        :return: a list of remaining regions
        """

        # if there is no overlap return the original region
        if (self.end is not None and self.end <= new_region.start) or\
                (new_region.end is not None and new_region.end <= self.start):
            return [self]

        # if the new region overlaps completely with this one.
        if new_region.start <= self.start and\
                ((new_region.end is None and self.end is None) or new_region.end is None or (self.end is not None and new_region.end >= self.end)):
            # print("Carve returning empty set")
            return []

        # if the new region overlaps with the start of this region
        if new_region.start <= self.start and\
                (self.end is None or (new_region.end < self.end)):
            # print("Carve shortening the start of this region")
            return [MemoryRegion(new_region.end, self.end)]

        # if the new region overlaps with the end of this region
        if self.end is not None and new_region.start < self.end and new_region.end >= self.end:
            # print("Carve shortening the end of this region")
            return [MemoryRegion(self.start, new_region.start)]

        # The only option left now is that the new rigion bisects this one, so return the two parts
        # print("Carve bisecting this region")
        return [MemoryRegion(self.start, new_region.start),
                MemoryRegion(new_region.end, self.end)]

    def can_fit_inside(self, super_region):
        """
        returns true if this region can fit inside the super_region
        :param super_region:
        :return: boolean
        """
        # if the super region is infinite then this is always true
        if super_region.end is None:
            return True

        # this region is infinite then this must be false
        if self.end is None:
            return False

        this_size = self.end - self.start
        super_size = super_region.end - super_region.start
        return this_size <= super_size


class MemoryRequirements:

    def __init__(self):
        self.blocks = []
        self.min_bound_blocks = []
        self.lower_bound = None

    def print_requirements(self):
        print("\n%d allocated tensor blocks used by %d operations\n" %
              (len(self.blocks),
               self.get_operation_count()))

        for i, b in enumerate(self.blocks):
            print("Block [%02d] size (%10d bytes) ops %3s - %3s" %
                  (i,
                   b.size,
                   b.creation,
                   b.last_use))

    def print_solution(self):
        print("print_solution ToDo")

    def get_operation_count(self):
        max_op = 0
        for b in self.blocks:
            max_op = max(max_op, b.creation, b.last_use)
        return max_op

    def calculate_lower_bound(self):
        max_concurrent_mem = 0

        # find the lower bound of required memory
        for op in range(self.get_operation_count()):
            concurrent_mem = 0
            for b in self.blocks:
                if b.overlaps(op):
                    concurrent_mem += b.size
            max_concurrent_mem = max(max_concurrent_mem, concurrent_mem)

        # mark all tensors which are part of the lower bound
        for op in range(self.get_operation_count()):
            concurrent_mem = 0
            for b in self.blocks:
                if b.overlaps(op):
                    concurrent_mem += b.size

            if concurrent_mem == max_concurrent_mem:
                for i, b in enumerate(self.blocks):
                    if b.overlaps(op):
                        self.min_bound_blocks[i] = True

        self.lower_bound = max_concurrent_mem

    @staticmethod
    def rect(img, left, top, right, bottom, color, border=1):

        if right < left:
            return
        if right == left:
            right = left+1

        img[left:right, top:bottom, :] = [0, 0, 0]

        left += border
        right -= border
        top += border
        bottom -= border

        if right > left and bottom > top:
            img[left+border:right-border, top+border:bottom-border, :] = color

    def save_memory_layout_image(self, blocks, file_name="memory.png"):

        memory_size = MemoryRequirements.required_memory(blocks)

        if memory_size == 0:
            print("Error: Cannot save memory layout when no blocks are allocated")
        else:
            row_height = 5

            img_width = row_height * self.get_operation_count()
            img_height = img_width

            img = np.zeros(shape=[img_height, img_width, 3], dtype=np.int8)

            # draw zebra background
            for i in range(self.get_operation_count()):
                if (i % 2) == 0:
                    img[(i*row_height):(i+1)*row_height, :, :] = [245, 245, 245]
                else:
                    img[(i*row_height):(i+1)*row_height, :, :] = [235, 235, 235]

            for i, b in enumerate(blocks):
                if b.allocated():
                    # print("Rendering allocated block with mem_offset %d" % b.mem_offset)
                    mem_start = int((b.mem_offset * img_width) / memory_size)
                    mem_end = int(((b.mem_offset + b.size) * img_width) / memory_size)

                    op_start = b.creation
                    op_end = b.last_use + 1

                    block_color = [255, 128, 128]
                    if self.min_bound_blocks[i]:
                        block_color = [128, 255, 128]

                    MemoryRequirements.rect(img,
                                            op_start*row_height, mem_start,
                                            op_end*row_height, mem_end,
                                            block_color)

            writer = png.Writer(width=img_width, height=img_height, bitdepth=8, greyscale=False)
            image_2d = np.reshape(img, (-1, img_width * 3))

            out = open(file_name, "wb")
            writer.write(out, image_2d)
            out.close()

    @staticmethod
    def required_memory(blocks):

        max_memory = 0
        for b in blocks:
            if b.allocated():
                max_memory = max(max_memory, b.mem_offset+b.size)

        return max_memory

    @staticmethod
    def unallocated_block_count(blocks):

        unallocated = 0
        for b in blocks:
            if not b.allocated():
                unallocated += 1

        return unallocated

    @staticmethod
    def heap_allocate_block(blocks, new_block_idx):
        """
        Add a single block to the allocated block pattern using
         a heap allocation method. I.e. the first free space.
        :param blocks:
        :param new_block_idx:
        :return: blocks with new block added in the first free location
        """

        # create a list of all free regions of memory around the blocks currently allocated
        # which overlap with this block
        free_regions = [MemoryRegion(0, None)]
        for b in blocks:
            if b.allocated() and b.overlaps(blocks[new_block_idx]):
                new_free_regions = []
                block_region = MemoryRegion(b.mem_offset, b.mem_offset+b.size)
                for region in free_regions:
                    new_free_regions += region.get_carve_result(block_region)
                free_regions = new_free_regions

        """print("About to add memory block of size %d bytes" % blocks[new_block_idx].size)
        print("%d Free areas are:" % len(free_regions))
        for i, f in enumerate(free_regions):
            print("[%2d] free from %d to %s" %
                  (i, f.start, f.end))"""

        # add this block to the first region it fits into
        new_block_region = MemoryRegion(0, blocks[new_block_idx].size)
        for region in free_regions:
            if new_block_region.can_fit_inside(region):
                blocks[new_block_idx].mem_offset = region.start
                break

        return blocks

    def ordered_heap_allocate(self, order):

        # allocate the memory blocks using a conventional heap allocation approach
        # using the given order

        # copy requirements and reset memory offsets
        heap_blocks = copy.deepcopy(self.blocks)
        for b in heap_blocks:
            b.mem_offset = None

        # allocate blocks in order
        for b_idx in order:
            heap_blocks = MemoryRequirements.heap_allocate_block(heap_blocks, b_idx)

        return heap_blocks

    def heap_allocation_method(self):

        # copy requirements and reset memory offsets
        heap_blocks = copy.deepcopy(self.blocks)
        for b in heap_blocks:
            b.mem_offset = None

        # create set of tensor indices ordered by creating operation
        ordered_tensors = []
        for op in range(self.get_operation_count()):
            for i, b in enumerate(self.blocks):
                if b.creation == op:
                    ordered_tensors += [i]

        print("Created an ordered list of block indices with %d elements" % len(ordered_tensors))

        # perform heap allocation strategy using this order
        return self.ordered_heap_allocate(ordered_tensors)

    def grow_from_list(self, inital_blocks):
        """
        Method to create an allocation pattern starting with an inital set of given blocks
        and then sequenntially adding the largest adjacent block using heap allocation.
        :param inital_blocks:
        :return:
        """

        # copy and reset block allocation pattern
        pattern = copy.deepcopy(self.blocks)
        for b in pattern:
            b.mem_offset = None

        # allocate the inital blocks in the order they've been given
        for b_idx in inital_blocks:
            pattern = MemoryRequirements.heap_allocate_block(pattern, b_idx)

        # while there are unallocated blocks find the largest block adjacent to existing allocated blocks
        # and allocate it
        while MemoryRequirements.unallocated_block_count(pattern) > 0:

            largest_adjacent_idx = None
            largest_adjacent_size = None

            for b_idx, b in enumerate(pattern):
                if not b.allocated():
                    if largest_adjacent_size is None or b.size > largest_adjacent_size:

                        overlaps = False
                        for ab in pattern:
                            if ab.allocated() and ab.overlaps(b):
                                overlaps = True
                                break

                        if overlaps:
                            largest_adjacent_idx = b_idx
                            largest_adjacent_size = b.size

            # if no largest adjacent block was found then there must be isolated regions of tensors which should
            # be impossible!
            if largest_adjacent_idx is None:
                print("grow_from_list Error: Couldn't find a largest adjacent block while some blocks are unallocated!")
            else:
                pattern = MemoryRequirements.heap_allocate_block(pattern, largest_adjacent_idx)

        return pattern

    def find_permutations_and_grow(self, unordered_blocks=[], ordered_blocks=[]):

        if len(unordered_blocks) == 0:
            return self.grow_from_list(ordered_blocks)
        else:
            # swap recurrse by swapping each element from the unordered set onto the end of the ordered list
            best_pattern = None
            best_mem_requirement = None
            for i in range(len(unordered_blocks)):
                new_ordered_blocks = ordered_blocks + [unordered_blocks[i]]
                # print("unordered_blocks is (%s) type %s" % (unordered_blocks, str(type(unordered_blocks))))
                new_unordered_blocks = unordered_blocks[:i] + unordered_blocks[i+1:]
                # print("new_unordered_blocks is (%s) type %s" % (new_unordered_blocks, str(type(new_unordered_blocks))))
                block_pattern = self.find_permutations_and_grow(new_unordered_blocks, new_ordered_blocks)

                if best_pattern is None:
                    best_pattern = block_pattern
                    best_mem_requirement = MemoryRequirements.required_memory(block_pattern)
                else:
                    mem_requirement = MemoryRequirements.required_memory(block_pattern)
                    if mem_requirement < best_mem_requirement:
                        best_pattern = block_pattern
                        best_mem_requirement = mem_requirement

                if best_mem_requirement == self.lower_bound:
                    print("returning pattern at lower_bound")
                    return best_pattern

            # if no pattern matching the lower bound was found then return the best
            print("returning best pattern after trying each permutation of the lower bound blocks")
            return best_pattern

    def lbb_growth_method(self):

        # find the first set of blocks which form the lower bound
        # NOTE it's possible there are several equal sized sets of blocks which co-define the lower bound
        low_bound_blocks = []
        for op in range(self.get_operation_count()):
            concurrent_mem = 0
            for b in self.blocks:
                if b.overlaps(op):
                    concurrent_mem += b.size
            if concurrent_mem == self.lower_bound:
                for i, b in enumerate(self.blocks):
                    if b.overlaps(op):
                        low_bound_blocks += [i]

        # use a recurrsive function to generate all possible orders of this set of blocks
        # WARNING n! runtime, so it checks that there are no more than
        # this implementation simply checks there are no more than 8 blocks in this set to limit the
        # runtime to 40k iterations of the growth algorithm
        lbbg_blocks = self.find_permutations_and_grow(unordered_blocks=low_bound_blocks)

        return lbbg_blocks

    def optimise(self):

        self.min_bound_blocks = [False] * len(self.blocks)

        print("\nOptimising memory use of %d block used by %d operations." %
              (len(self.blocks),
               self.get_operation_count()))

        self.calculate_lower_bound()
        print("\nCalculated a lower bound of %d bytes" % self.lower_bound)

        # find upper bound using naive head allocation method
        heap_allocated_blocks = self.heap_allocation_method()
        upper_bound = MemoryRequirements.required_memory(heap_allocated_blocks)
        print("\nCalculated an upper bound of %d bytes using the heap allocation strategy" % upper_bound)
        self.save_memory_layout_image(heap_allocated_blocks, file_name="heap_mem.png")

        # optimise using growing from lower_bound_blocks method
        lbb_growth_blocks = self.lbb_growth_method()
        lbb_size = MemoryRequirements.required_memory(lbb_growth_blocks)
        print("\nCalculated an optimised memory size of %d bytes using the lbb_growth strategy" % lbb_size)
        self.save_memory_layout_image(lbb_growth_blocks, file_name="lbbg_mem.png")

        # display the results of each starting state
