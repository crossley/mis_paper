library(data.table)
library(ggplot2)

d <- fread('../data/MIS_DATA_LONG_17122019.csv')

d[, subject := Subject]

d[phase == 'Adaptation', trial := trial + 198]
d[phase == 'Generalisation', trial := trial + 198 + 110]

dd <- d[, mean(hand_angle), .(group, subject, phase, trial, target)]

ggplot(dd, aes(trial, V1, colour=as.factor(target))) +
    geom_point() +
    facet_wrap(~group*subject)
