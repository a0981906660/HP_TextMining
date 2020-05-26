rm -rf ./out/*
rm -rf ./out.csv

java -cp "opinion/*.jar:." CopeOpi_simp file_simp.lst -d

