from LDAGibbs import LDAGibbs
import codecs
import os


def read_voca_file(file_path):
    """
    Read vocabulary file
    :param file_path: The path of vocabulary file
    :return: vocabulary list
    """
    vocas = list()

    with codecs.open(file_path, "r", "utf-8") as voca_file:
        for each_line in voca_file:
            vocas.append(each_line.strip())

    return vocas


def main():
    document_file_path = "./data/ap.dat"
    voca_file_path = "./data/vocab.txt"
    output_dir_name = "./output_LDA_Gibbs/"
    topics = 50
    iterations = 20

    vocas = read_voca_file(voca_file_path)

    os.makedirs(output_dir_name, exist_ok=True)

    LDA_Model = LDAGibbs(topics, document_file_path, vocas, output_dir_name)
    LDA_Model.run(max_iter=iterations, do_print_log=True)
    LDA_Model.export_result("Complete")


if __name__ == '__main__':
    main()
