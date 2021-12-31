# Install and activate tidyverse - a bundle of useful packages
install.packages('tidyverse')
library(tidyverse)

# Load data
graded_readers <- read_csv('graded_readers_stats.csv')

columns <- c(
  'Occurrence count (Inicial)',
  'Occurrence count (Intermedio)',
  'Occurrence count (Avanzado)',
  'Occurrence count (Infantil)',
  'Occurrence count (Juvenil)',
  'Occurrence count (Adulta)',
  'Occurrence count (Native corpus)'
)

mask_1 <- graded_readers['Level'] == 'A1-A2'
mask_2 <- graded_readers['Level'] == 'B1'
mask_3 <- graded_readers['Level'] == 'B2'

# graded_readers[mask_2, ] & graded_readers['Occurrence count (Inicial)'] > 0
# head(graded_readers[mask_2, ]['Occurrence count (Inicial)'] > 0)
tail(graded_readers[mask_1, ]['Lexical item'])
tail(graded_readers[mask_1, ]['Occurrence count (Inicial)'])
tail(graded_readers[mask_1, ]['Occurrence count (Inicial)'] > 0)

for (i in 1:length(columns)) {
  occurrences <- sum(graded_readers[columns[i]] > 0)
  total <- nrow(graded_readers)
  # print(paste(columns[i], occurrences / total, sep = ': '))
  print(occurrences / total)
}
