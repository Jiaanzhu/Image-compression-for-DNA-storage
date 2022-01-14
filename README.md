# Image-compression-for-DNA-storage

## Main information
1.The original README provided by Xavier Pic(the Mediacoding group in the I3S laboratory, Université Côte d’Azur) is the file **Xavier_README.md**, the information about installation and commends are included.

2.The main code of this project is taken from https://github.com/jpegdna-mediacoding/Jpeg_DNA_Python (the version updated on 14 December, 2021). Some code modifications are made to avoid decoding script stopping due to 'IndexError: list index out of range’. 

3.The script I used is in the folder **jpegdna/scripts**, including:

**jpegdnargb_eval.py** 
Input: original image, Output: the compressed image file(in png format), DNA sequence file and info file(about size of image, huffman tables).

**Analysis.py** 
Input: DNA sequence file, info file, and original image as input, Output: image quality analysis plots.

**mesa_eval.py** 
Input: DNA sequence file, Output: DNA sequence file(Mutated). Mutate the sequence by MESA(https://github.com/umr-ds/mesa_dna_sim) in an API way.



**convert_fasta.py** 
Input: DNA sequence file, Output: DNA sequence fasta file. (Some DNA error simulator require the input in fasta format, for example DeepSimulator: https://github.com/liyu95/DeepSimulator)



