# A generic binary heap backed by an Array.
# *T* must include Comparable(T).
#
# Pass a block to `new` to control ordering:
#   BinaryHeap(Candidate).new { |a, b| a.dist <= b.dist }  # min-heap (nearest first)
#   BinaryHeap(Candidate).new { |a, b| a.dist >= b.dist }  # max-heap (farthest first)
#
# Operations:
#   push(item)   O(log n)
#   pop          O(log n) — removes and returns the root
#   peek         O(1)     — returns root without removing
#   size / empty?
module Vecstolite
  class BinaryHeap(T)
    @data : Array(T)
    @cmp : Proc(T, T, Bool)

    def initialize(&block : T, T -> Bool)
      @data = [] of T
      @cmp = block
    end

    def push(item : T) : Nil
      @data << item
      sift_up(@data.size - 1)
    end

    def pop : T
      raise "Heap is empty" if @data.empty?
      top = @data[0]
      last = @data.pop
      unless @data.empty?
        @data[0] = last
        sift_down(0)
      end
      top
    end

    def peek : T
      raise "Heap is empty" if @data.empty?
      @data[0]
    end

    def size : Int32
      @data.size
    end

    def empty? : Bool
      @data.empty?
    end

    private def sift_up(i : Int32) : Nil
      while i > 0
        parent = (i - 1) >> 1
        break if @cmp.call(@data[parent], @data[i])
        @data.swap(parent, i)
        i = parent
      end
    end

    private def sift_down(i : Int32) : Nil
      n = @data.size
      loop do
        left = 2 * i + 1
        right = 2 * i + 2
        best = i
        best = left if left < n && @cmp.call(@data[left], @data[best])
        best = right if right < n && @cmp.call(@data[right], @data[best])
        break if best == i
        @data.swap(i, best)
        i = best
      end
    end
  end
end
