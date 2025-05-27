import os

files = ["corpus/tao.txt", "corpus/gita.txt", "corpus/analects.txt", "corpus/confucius.txt"]
with open("corpus/all_combined.txt", "w", encoding="utf-8") as out:
    for f in files:
        with open(f, encoding="utf-8") as infile:
            lines = infile.readlines()
            clean = [line.strip() for line in lines if line.strip()]
            out.write("\n".join(clean) + "\n")