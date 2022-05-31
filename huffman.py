from collections import Counter
import heapq

from bitarray import bitarray
import graphviz

import pandas as pd


import os
os.environ["PATH"] += os.pathsep + 'C:/Users/PC/anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz'
import filecmp
import time


class Huffman:
    def __init__(self):
        self.heap = []
        self.codes = {}


    class HeapNode:
        """Class for building the pririty queue. Takes character, frequency left and right. Returns compared values,
        when building the priority queue"""

        def __init__(self, character, frequency=0, left=None, right=None):
            self.character = character
            self.frequency = frequency
            self.left = left
            self.right = right

        def __lt__(self, other):
            return self.frequency < other.frequency

    def make_heap(self, frequency):
        """function which is pushing heaps from class HeapNode to heap list"""
        for f in frequency:
            node = self.HeapNode(f, frequency[f])
            heapq.heappush(self.heap, node)

    def get_character_frequency(self, text):
        """Function that takes text as input and returns all unique characters in dictionary in ascending order"""
        unique_chars_freq = Counter([chars for chars in text])
        unique_chars_freq = dict(sorted(unique_chars_freq.items(), key=lambda item: item[1]))
        return unique_chars_freq

    def build_huffman_tree(self):
        """Function that build the Huffman tree. Merges all the nodes into one and return one node value"""
        while len(self.heap) > 1:
            left = heapq.heappop(self.heap)
            right = heapq.heappop(self.heap)
            root = self.HeapNode(None, frequency=left.frequency + right.frequency)
            root.left = left
            root.right = right
            heapq.heappush(self.heap, root)
        huffman_tree = heapq.heappop(self.heap)
        return huffman_tree

    def huffman_tree_encoding(self, tree, current_code):
        """Function that encodes the Huffman tree and returns its binary value"""
        if tree.character is not None:
            current_code += "1"
            current_code += f"{ord(tree.character):08b}"
        else:
            current_code += "0"
            current_code = self.huffman_tree_encoding(tree.left, current_code)
            current_code = self.huffman_tree_encoding(tree.right, current_code)

        return current_code

    def make_binary_codes_helper(self, tree, current_code):
        """A helper function for make_binary_codes_function. Adds binary values of characters to dictionary"""
        if tree is None:
            return

        if tree.character is not None:
            self.codes[tree.character] = current_code
            return

        self.make_binary_codes_helper(tree.left, current_code + "0")
        self.make_binary_codes_helper(tree.right, current_code + "1")

    def make_binary_codes(self, huffman_tree):
        """A function that calls make_codes_helper and huffman_tree_encoding.
        Returns the encoded tree"""

        current_code = ""

        self.make_binary_codes_helper(huffman_tree, current_code)

    def get_encoded_text(self, text):
        """Function that takes text as param and returns encoded text"""
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]

        return encoded_text

    def combine(self, encoded_text, encoded_tree):
        """Function combining the encoded_tree and the encoded_text.
        Adds padding if needed.
        Returns a string of combined elements"""

        padding_number = 8 - (len(encoded_text) + len(encoded_tree)) % 8
        if padding_number != 0:
            encoded_text = padding_number * "0" + encoded_text

        return f"{encoded_tree}{padding_number:08b}{encoded_text}"

    def compress(self,path):
        """Save encoded text to output file"""
        start_time = time.perf_counter ()
        
        f, filename_encoding = os.path.splitext(path)
        filename = os.path.basename(path)
        output_path = "compressed/" + filename.replace(filename_encoding,"") + ".bin"

        with open(path,encoding="utf-8") as in_file, open(output_path, "wb") as out_file:
            text = in_file.read()

            encoding, tree, freq = self.run_compress(text)

            byte_array = bytearray()
            for i in range(0, len(encoding), 8):
                byte_array.append(int(encoding[i:i + 8], 2))

            out_file.write(byte_array)

            end_time = time.perf_counter ()

        total_time_compress = end_time - start_time

        return tree, freq, total_time_compress, self.codes

    def run_compress(self, text):
        """Helper function loading for compress"""
        freq = self.get_character_frequency(text)
        self.make_heap(freq)

        tree = self.build_huffman_tree()

        encoded_tree = self.huffman_tree_encoding(tree, "")

        self.make_binary_codes(tree)

        encoded_text = self.get_encoded_text(text)

        encoded_text_and_header = self.combine(encoded_text, encoded_tree)

        return encoded_text_and_header, tree, freq

    def decompress(self,path):
        """Save decoded text to output file"""
        start_time = time.perf_counter ()
        
        f, filename_encoding = os.path.splitext(path)
        filename = os.path.basename(path)
        output_path = "decompressed/" + filename.replace(filename_encoding,"") + "_decompressed" + ".txt"

        with open(path, "rb") as in_file, open(output_path, "w") as out_file:
            encoded_text = ""

            byte = in_file.read(1)
            while len(byte) > 0:
                encoded_text += f"{bin(ord(byte))[2:]:0>8}"
                byte = in_file.read(1)

            decoded_text = self.decode(encoded_text)
            out_file.write(decoded_text)

        end_time = time.perf_counter ()
        total_time_decompress = end_time - start_time
        return total_time_decompress

    def huffman_tree_decoding(self, tree_code):
        """Decoding huffman tree to be able to decode the encoded text"""
        code_bit = tree_code[0]
        del tree_code[0]

        if code_bit == "1":
            char = ""
            for _ in range(8):
                char += tree_code[0]
                del tree_code[0]

            return self.HeapNode(chr(int(char, 2)))

        return self.HeapNode(None, left=self.huffman_tree_decoding(tree_code), right=self.huffman_tree_decoding(tree_code))

    def remove_padding(self, encoded_text_with_padding):

        number_of_extra_zeros_bin = encoded_text_with_padding[:8]
        encoded_text_with_padding = encoded_text_with_padding[8:]
        number_of_extra_zeros = int("".join(number_of_extra_zeros_bin), 2)
        encoded_text_no_padding = encoded_text_with_padding[number_of_extra_zeros:]

        return encoded_text_no_padding

    def decode(self, encoded_text):
        """Returns decoded string"""

        encoded_text_list = list(encoded_text)
        encoded_tree = self.huffman_tree_decoding(encoded_text_list)

        encoded_tect_no_padding = self.remove_padding(encoded_text_list)

        text = ""
        current_node = encoded_tree
        for char in encoded_tect_no_padding:
            current_node = current_node.left if char == '0' else current_node.right

            if current_node.character is not None:
                text += current_node.character
                current_node = encoded_tree

        return text

    def test(self,original_file_path,decompressed_file_path):
        """Checks if original file is the same as decompressed."""
        check_original_decompressed_file = filecmp.cmp(original_file_path, decompressed_file_path)
        return check_original_decompressed_file

    def visualize(self, tree):
        """A function that visualizes the prebuild Huffman tree"""
        dot = graphviz.Digraph(name='Huffman Tree',node_attr={"fontsize":"5","fixedsize":"True","width":"0.3","height":"0.3"})

        def postorder(tree):
            if tree:
                postorder(tree.left)
                postorder(tree.right)
                dot.node(name=str(tree.frequency) + str(tree.character), label=str(tree.frequency) + '\n' + str(tree.character))
                if tree.left:
                    dot.edge(str(tree.frequency) + str(tree.character), str(tree.left.frequency) + str(tree.left.character))
                if tree.right:
                    dot.edge(str(tree.frequency) + str(tree.character), str(tree.right.frequency) + str(tree.right.character))

        postorder(tree)
        # render in script
#         dot.render('huffman tree', view=True)
        return dot

    def huffman_table(self, codes, freq):
        """A function that takes the codes and character frequency and returns pandas table with
    values, codes and frequencies"""
        table = pd.DataFrame.from_dict([codes])
        table = table.transpose()
        table = table.reset_index()
        table.columns = ["Value", "Code"]
        frequency = []
        for i in table["Value"]:
            for k, v in freq.items():
                if i == k:
                    frequency.append(v)

        table["Frequency"] = frequency
        table = table.sort_values(by="Frequency")
        table.reset_index()
        return table

    def size_comparison(self, compressed_file_path, original_file_path):
        """Compares the sizes of two files. Returns sizes in bytes"""
        size_compressed_file = os.path.getsize(compressed_file_path)
        size_original_file = os.path.getsize(original_file_path)

        return f"Compressed file size : {size_compressed_file} bytes" \
               f"Uncompressed file size :  {size_original_file} bytes"

    def compression_percentage_and_ratio(self,compressed_file_path, original_file_path):   
        """Takes original file and compressed file sizes. Returns compression percentage and ratio."""

        size_compressed_file = int(os.path.getsize(compressed_file_path))
        size_original_file = int(os.path.getsize(original_file_path))

        percentage_compression = 100 - (size_compressed_file / size_original_file * 100)
        ratio = size_compressed_file / size_original_file

        return print(f"Compression percentage : {percentage_compression:.2f}% \n" \
                 f"Compression ratio :  {ratio:.2f}\n")

    def compression_time(self,compress_time):
        """Returns compression time."""
        return f"Compression time : {compress_time:.4f} seconds."

    def decompression_time(self,decompress_time):
        """Returns decompression time."""
        return f"Decompression time : {decompress_time:.4f} seconds."


