"""Test module for the formatting"""

import numpy as np

from jpegdna.format import JpegDNAFormatter
from jpegdna.format import GeneralInfoFormatter
from jpegdna.format import GrayFrequenciesFormatter, RGBFrequenciesFormatter
from jpegdna.format import DataFormatter

def generalinfoformatter_test():
    """Functionnal tests for the gray level general info formatter"""
    alpha = 0.1756
    freq_origin = "from_img"
    m, n = 512, 512
    blockdims = (8, 8)
    max_cat = 11
    max_runcat = 162
    dc_freq_len = 7
    ac_freq_len = 10
    header = "ATCGATC"
    image_type = "gray"
    sampler = None
    formatter = GeneralInfoFormatter(alpha,
                                     freq_origin,
                                     m,
                                     n,
                                     blockdims,
                                     max_cat,
                                     max_runcat,
                                     dc_freq_len,
                                     ac_freq_len,
                                     image_type,
                                     sampler,
                                     header)
    oligo = formatter.format(None)
    print(oligo)
    formatter = GeneralInfoFormatter(None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     header)
    formatter.deformat(oligo[7:])
    assert round(formatter.alpha, 3) == round(alpha, 3)
    assert formatter.blockdims == blockdims
    assert formatter.n == n
    assert formatter.m == m
    assert formatter.freq_origin == freq_origin
    assert formatter.max_cat == max_cat
    assert formatter.max_runcat == max_runcat
    assert formatter.dc_freq_len == dc_freq_len
    assert formatter.ac_freq_len == ac_freq_len

def rgb_generalinfoformatter_test():
    """Functionnal tests for the RGB general info formatter"""
    alpha = 0.1756
    freq_origin = "from_img"
    m, n = 512, 512
    blockdims = (8, 8)
    max_cat = 11
    max_runcat = 162
    dc_freq_len = 7
    ac_freq_len = 10
    image_type = "RGB"
    header = "ATCGATC"
    sampler = "4:2:2"
    formatter = GeneralInfoFormatter(alpha,
                                     freq_origin,
                                     m,
                                     n,
                                     blockdims,
                                     max_cat,
                                     max_runcat,
                                     dc_freq_len,
                                     ac_freq_len,
                                     image_type,
                                     sampler,
                                     header)
    oligo = formatter.format(None)
    print(oligo)
    formatter = GeneralInfoFormatter(None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     "4:2:2",
                                     header)
    formatter.deformat(oligo[7:])
    assert round(formatter.alpha, 3) == round(alpha, 3)
    assert formatter.blockdims == blockdims
    assert formatter.n == (n, 512, 512)
    assert formatter.m == (m, m/2, m/2)
    assert formatter.freq_origin == freq_origin
    assert formatter.max_cat == max_cat
    assert formatter.max_runcat == max_runcat
    assert formatter.dc_freq_len == dc_freq_len
    assert formatter.ac_freq_len == ac_freq_len
    assert formatter.sampler == sampler

def frequenciesformatter_test():
    """Functionnal tests for the gray level frequencies formatter"""
    max_cat = 11
    max_runcat = 162
    dc_freq_len = 7
    ac_freq_len = 10
    header = "ATCGATC"
    dc_header = "ATCG"
    ac_header = "TCGA"
    formatter = GrayFrequenciesFormatter(max_cat, max_runcat, dc_freq_len, ac_freq_len, header, dc_header, ac_header)
    ac_freqs = np.zeros((162)).astype(int)
    dc_freqs = np.zeros((11)).astype(int)
    oligos = formatter.format((dc_freqs, ac_freqs))
    oligos_in = []
    for oligo in oligos:
        oligos_in.append(oligo[7:])
    (dc_freqs_out, ac_freqs_out) = formatter.deformat(oligos_in)
    assert (dc_freqs_out == dc_freqs).all()
    assert (ac_freqs_out == ac_freqs).all()
    formatter = GrayFrequenciesFormatter(max_cat, max_runcat, dc_freq_len, ac_freq_len, header, dc_header, ac_header, debug=True)
    ac_freqs = np.zeros((162)).astype(int)
    dc_freqs = np.zeros((11)).astype(int)
    oligos = formatter.format((dc_freqs, ac_freqs))

def rgb_frequenciesformatter_test():
    """Functionnal tests for the RGB frequencies formatter"""
    max_cat = 11*3
    max_runcat = 162*3
    dc_freq_len = 7
    ac_freq_len = 10
    header = "ATCGATC"
    dc_header = "ATCG"
    ac_header = "TCGA"
    formatter = RGBFrequenciesFormatter(max_cat, max_runcat, dc_freq_len, ac_freq_len, header, dc_header, ac_header)
    ac_freqs = (np.zeros((162)).astype(int), np.ones((162)).astype(int), np.zeros((162)).astype(int))
    dc_freqs = (np.zeros((11)).astype(int), np.ones((11)).astype(int), np.zeros((11)).astype(int))
    oligos = formatter.format((dc_freqs, ac_freqs))
    oligos_in = []
    for oligo in oligos:
        oligos_in.append(oligo[7:])
    (dc_freqs_out, ac_freqs_out) = formatter.deformat(oligos_in)
    for i in range(3):
        assert (dc_freqs_out[i] == dc_freqs[i]).all()
        assert (ac_freqs_out[i] == ac_freqs[i]).all()
    formatter = RGBFrequenciesFormatter(max_cat, max_runcat, dc_freq_len, ac_freq_len, header, dc_header, ac_header, debug=True)
    ac_freqs = (np.zeros((162)).astype(int), np.ones((162)).astype(int), np.zeros((162)).astype(int))
    dc_freqs = (np.zeros((11)).astype(int), np.ones((11)).astype(int), np.zeros((11)).astype(int))
    oligos = formatter.format((dc_freqs, ac_freqs))

def dataformatter_test():
    """Functionnal tests for the data formatter"""
    formatter = DataFormatter("ATCGATC")
    length = 500
    word = "ATCG"
    data_strand = word*length
    oligos = formatter.format(data_strand)
    deformat_oligos = []
    for oligo in oligos:
        deformat_oligos.append(oligo[7:])
    data_strand_out = formatter.deformat(deformat_oligos)
    assert data_strand == data_strand_out[:length*len(word)]
    formatter = DataFormatter("ATCGATC", debug=True)
    length = 500
    word = "ATCG"
    data_strand = word*length
    oligos = formatter.format(data_strand)

def jpegdna_test():
    """Functionnal tests for the general gray level jpegdna formatter"""
    image_strand = "ATCG" * 2000
    debug = False
    choice = "from_img"
    formatter = JpegDNAFormatter(1, "gray", primer="illumina", oligo_length=200, debug=debug)
    oligos = formatter.full_format(image_strand, choice, 512, 512, np.zeros((11)).astype(int), np.zeros((162)).astype(int))
    data_strand, (alpha, m, n, freq_dc, freq_ac) = formatter.full_deformat(oligos)
    assert data_strand[:len(image_strand)] == image_strand
    assert alpha == 1
    assert m == 512
    assert n == 512
    assert (freq_dc == np.zeros((11)).astype(int)).all()
    assert (freq_ac == np.zeros((162)).astype(int)).all()
    choice = "default"
    formatter = JpegDNAFormatter(1, "gray", primer="illumina", oligo_length=200, debug=debug)
    oligos = formatter.full_format(image_strand, choice, 512, 512, np.zeros((11)).astype(int), np.zeros((162)).astype(int))
    data_strand, (alpha, m, n, freq_dc, freq_ac) = formatter.full_deformat(oligos)
    assert data_strand[:len(image_strand)] == image_strand
    assert alpha == 1
    assert m == 512
    assert n == 512
    assert freq_dc is None
    assert freq_ac is None
    debug = True
    choice = "from_img"
    formatter = JpegDNAFormatter(1, "gray", primer="illumina", oligo_length=200, debug=debug)
    oligos = formatter.full_format(image_strand, choice, 512, 512, np.zeros((11)).astype(int), np.zeros((162)).astype(int))
    data_strand, (alpha, m, n, freq_dc, freq_ac) = formatter.full_deformat(oligos)
    choice = "default"
    formatter = JpegDNAFormatter(1, "gray", primer="illumina", oligo_length=200, debug=debug)
    oligos = formatter.full_format(image_strand, choice, 512, 512, np.zeros((11)).astype(int), np.zeros((162)).astype(int))
    data_strand, (alpha, m, n, freq_dc, freq_ac) = formatter.full_deformat(oligos)

def jpegdnargb_test():
    """Functionnal tests for the general rgb jpegdna formatter"""
    image_strand = "ATCG" * 2000
    debug = False
    choice = "from_img"
    formatter = JpegDNAFormatter(1, "RGB", primer="illumina", oligo_length=200, debug=debug)
    oligos = formatter.full_format(image_strand,
                                   choice,
                                   512,
                                   512,
                                   (np.zeros((11)).astype(int), np.zeros((11)).astype(int), np.zeros((11)).astype(int)),
                                   (np.zeros((162)).astype(int), np.zeros((162)).astype(int), np.zeros((162)).astype(int)))
    data_strand, (alpha, m, n, freq_dc, freq_ac) = formatter.full_deformat(oligos)
    assert data_strand[:len(image_strand)] == image_strand
    assert alpha == 1
    assert m == (512, 256, 256)
    assert n == (512, 512, 512)
    assert (freq_dc == np.zeros((11)).astype(int)).all()
    assert (freq_ac == np.zeros((162)).astype(int)).all()
    choice = "default"
    formatter = JpegDNAFormatter(1, "RGB", primer="illumina", oligo_length=200, debug=debug)
    oligos = formatter.full_format(image_strand,
                                   choice,
                                   512,
                                   512,
                                   (np.zeros((11)).astype(int), np.zeros((11)).astype(int), np.zeros((11)).astype(int)),
                                   (np.zeros((162)).astype(int), np.zeros((162)).astype(int), np.zeros((162)).astype(int)))
    data_strand, (alpha, m, n, freq_dc, freq_ac) = formatter.full_deformat(oligos)
    assert data_strand[:len(image_strand)] == image_strand
    assert alpha == 1
    assert m == (512, 256, 256)
    assert n == (512, 512, 512)
    assert freq_dc == (None, None, None)
    assert freq_ac == (None, None, None)
    debug = True
    choice = "from_img"
    formatter = JpegDNAFormatter(1, "RGB", primer="illumina", oligo_length=200, debug=debug)
    oligos = formatter.full_format(image_strand,
                                   choice,
                                   512,
                                   512,
                                   (np.zeros((11)).astype(int), np.zeros((11)).astype(int), np.zeros((11)).astype(int)),
                                   (np.zeros((162)).astype(int), np.zeros((162)).astype(int), np.zeros((162)).astype(int)))
    data_strand, (alpha, m, n, freq_dc, freq_ac) = formatter.full_deformat(oligos)
    choice = "default"
    formatter = JpegDNAFormatter(1, "RGB", primer="illumina", oligo_length=200, debug=debug)
    oligos = formatter.full_format(image_strand,
                                   choice,
                                   512,
                                   512,
                                   (np.zeros((11)).astype(int), np.zeros((11)).astype(int), np.zeros((11)).astype(int)),
                                   (np.zeros((162)).astype(int), np.zeros((162)).astype(int), np.zeros((162)).astype(int)))
    data_strand, (alpha, m, n, freq_dc, freq_ac) = formatter.full_deformat(oligos)
