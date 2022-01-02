# Install and activate tidyverse - a bundle of useful packages
install.packages('tidyverse')
library(tidyverse)

# Load data
graded_readers <- read_csv('graded_readers_stats.csv', n_max = 5)

###############################################################################
# Split tree tuples into columns
###############################################################################

columns <- c(
  "Trees' min, max, and average width & depth (Inicial)",
  "Trees' min, max, and average width & depth (Intermedio)",
  "Trees' min, max, and average width & depth (Avanzado)",
  "Trees' min, max, and average width & depth (Infantil)",
  "Trees' min, max, and average width & depth (Juvenil)",
  "Trees' min, max, and average width & depth (Adulta)",
  "Trees' min, max, and average width & depth (Native corpus)"
)

new_columns <- c(
  "Tree_Inicial",
  "Tree_Intermedio",
  "Tree_Avanzado",
  "Tree_Infantil",
  "Tree_Juvenil",
  "Tree_Adulta",
  "Tree_Native"
)

for (i in 1:length(columns)) {
  current <- columns[i]
  new <- new_columns[i]
  colnames(graded_readers)[which(names(graded_readers) == current)] <- new
}

strip_parens <- function(x) print(gsub('[()]', '', x))

# Remove ( and ) from start and end of each tuple
for (i in 1:length(new_columns)) {
  graded_readers <- data.frame(apply(graded_readers, 2, strip_parens))
}

for (i in 1:length(new_columns)) {
  current <- new_columns[i]
  index <- which(names(graded_readers) == current)
  new_cols <- c(
    paste(current, 'MinW', sep = '_'),
    paste(current, 'MaxW', sep = '_'),
    paste(current, 'AvgW', sep = '_'),
    paste(current, 'MinH', sep = '_'),
    paste(current, 'MaxH', sep = '_'),
    paste(current, 'AvgH', sep = '_')
  )
  graded_readers <- graded_readers %>% separate(index, new_cols, ', ', 
                                                remove = FALSE)
}

# FOR DEBUGGING ONLY
# Delete all columns before Tree Properties to avoid horizontal scrolling
graded_readers[1:58] <- NULL




###############################################################################
#                               Commented Code                                #
###############################################################################

# columns <- c(
#   'Occurrence count (Inicial)',
#   'Occurrence count (Intermedio)',
#   'Occurrence count (Avanzado)',
#   'Occurrence count (Infantil)',
#   'Occurrence count (Juvenil)',
#   'Occurrence count (Adulta)',
#   'Occurrence count (Native corpus)'
# )
# 
# mask_1 <- graded_readers['Level'] == 'A1-A2'
# mask_2 <- graded_readers['Level'] == 'B1'
# mask_3 <- graded_readers['Level'] == 'B2'
# 
# # graded_readers[mask_2, ] & graded_readers['Occurrence count (Inicial)'] > 0
# # head(graded_readers[mask_2, ]['Occurrence count (Inicial)'] > 0)
# tail(graded_readers[mask_1, ]['Lexical item'])
# tail(graded_readers[mask_1, ]['Occurrence count (Inicial)'])
# tail(graded_readers[mask_1, ]['Occurrence count (Inicial)'] > 0)
# 
# for (i in 1:length(columns)) {
#   occurrences <- sum(graded_readers[columns[i]] > 0)
#   total <- nrow(graded_readers)
#   # print(paste(columns[i], occurrences / total, sep = ': '))
#   print(occurrences / total)
# }







