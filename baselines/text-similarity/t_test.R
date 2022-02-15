setwd("<path>/bva-capstone/baselines/text-similarity/")
library(data.table)
library(stringr)
library(ggplot2)
library(scales)
library(dplyr)
options(show.error.locations = TRUE)

remove_na <- function(df) {
    df[is.na(df)] = ""
    return(df)
}

make_stars <- function(pval) {
    if (is.nan(pval)) {
    }
    else if (pval < 0.001) {
        return ("***")
    } else if (pval < 0.01) {
        return ("**")
    } else if (pval < 0.05) {
        return ("***")
    }
    return ("")
}

zip <- function(a, b, sep="_") {
    l = c()
    for (ai in a) {
        for (bi in b) {
            l = c(l , paste(ai, bi, sep=sep))
        }
    }
    return (l)
}

plot_mean <- function(results, xcol, ycol) {
    m1 = paste(ycol, "mean", sep="_")
    m2 = paste(ycol, "se", sep="_")
    measure_vars = c(m1, m2)
    df_mean = melt(results, id.vars=xcol, measure.vars=m1, value.name="mean")
    df_mean[, variable := gsub("_mean", "", variable)]
    df_se = melt(results, id.vars=xcol, measure.vars=m2, value.name="se")
    df_se[, variable := gsub("_se", "", variable)]
    results = merge(df_mean, df_se, by=c("variable", xcol))
    results[["ymin"]] = results$mean - 2*results$se
    results[["ymax"]] = results$mean + 2*results$se
    ggplot(results, aes_string(x=xcol, y="mean", col="variable")) +
        geom_linerange(aes(ymin=ymin, ymax=ymax), alpha=0.5, size=2) +
        geom_point(size=3)
}

# Use this function to decide which cols to include
get_colnames <- function(target="cit_idx", labels=c("_mean", "_se")) {
    # Get Metrics for Each
    if (target == "cit_idx") {
        metrics = c("recall_1", "recall_5", "recall_20")
        classes = c("")
    } else if (target == "cit_class") {
        metrics = c("F1", "P", "R", "Macro-F1")
        classes = c("cit_2", "cit_3", "cit_4")
    }

    # Get Column Names
    cols = c()
    temp_classes = classes
    for (metric in metrics) {
        if (metric == "Macro-F1") temp_classes = c("")
        for (cl in temp_classes) {
            col = cond_paste(cl, metric)
            for (label in labels) {
                cols = c(cols, paste0(col, label))
            }
        }
    }
    return (list("metrics"=metrics, "classes"=classes, "cols"=cols))
}

pretty_table <- function(filename, split_cols, labels=c("_mean", "_se")) {
    results <- fread(filename)
    results = remove_na(results)
    results[, run := do.call(paste, c(results[, ..split_cols], sep="-"))]
    headers <- unique(results[, .SD, .SDcols = c(split_cols, "run", "target")])
    runs = unique(results$run)
    target = results$target[1]
    n = as.integer(str_extract(results$partition[1], '\\b\\w+$'))

    l = get_colnames(target, labels=labels)
    metrics = l[["metrics"]]
    classes = l[["classes"]]
    cols = l[["cols"]]

    rows = list()
    for (i in 1:length(runs)) {
        row = list()
        prev = runs[i-1]
        curr = runs[i]
        row["run"] = curr

        temp_classes = classes
        for (metric in metrics) {    
            if (metric == "Macro-F1") temp_classes = c("")
            for (cl in temp_classes) {
                col = cond_paste(cl, metric)
                y = results[run == curr][[col]]
                row[paste0(col, "_mean")] = mean(y)
                row[paste0(col, "_se")] = sd(y) / sqrt(n)

                if (i > 1) {
                    x = results[run == prev][[col]]
                    test = t.test(x, y, alternative="two.sided", paired=TRUE)  
                    p = test$p.value
                    diff = -test$estimate
                    row[paste0(col, "_pval")] = make_stars(p)
                    row[paste0(col, "_diff")] = round(diff, 1)
                }
                row[paste0(col, "_mean")] = round(mean(y),1)
            }
        }      
        rows[[i]] = row
    }
    tests = rbindlist(rows, fill=TRUE)
    tests$run = runs
    tests = merge(tests, headers, by="run", sort=FALSE)
    cols = c("run", "target", cols, split_cols)
    tests = tests[, ..cols]
    return(tests)
}

print_overleaf <- function(results) {
    target = results$target[1]

    if (target == "cit_class") {
        cols = c("Macro-F1", "cit_2-F1", "cit_2-P", "cit_2-R",
                 "cit_3-F1", "cit_3-P", "cit_3-R", 
                 "cit_4-F1", "cit_4-P", "cit_4-R")
        cols = paste(cols, "mean", sep="_")
        results = results[, ..cols]
        names(results) = sub("_mean", "", names(results))
        for (col in names(results)) {
            results[[col]] = format(results[[col]], nsmall=1)
            results[[col]] = paste0(results[[col]], "\\%")
        }
    }

    if (target == "cit_idx") {
        cols = c("recall_1", "recall_5", "recall_20")
        labels = c("mean", "se")
        cols = zip(cols, labels, "_") 
        results = results[, ..cols]
        for (col in names(results)) {
            if (grepl("_mean", col)) {
                results[[col]] = format(results[[col]], nsmall=1)
                results[[col]] = paste0(results[[col]], "\\%")
            }
            if (grepl("_se", col)) {
                results[[col]] = format(round(results[[col]],2), nsmall=2)
                results[[col]] = paste0("(", results[[col]], "\\%)")
            }
        }
    }

    row_labels = c("& Original & ",
                   "& Original + Year & ",
                   "& Original + Year + Class & ",
                   "& Original + Year + Class + VLJ & ")

    message = ""
    for (i in 1:nrow(results)) {
        row = results[i]
        message = paste0(message, row_labels[i])
        message = paste0(message, paste(row, collapse=" & "))
        message = paste0(message, " \\\\")
        message = paste0(message, "\n")
    }
    cat(message)
}

# Expt: Vary cit count

results = pretty_table("results/cit_class_vary_cit_count.txt", 
                       split_cols=c("max_cit_count"))
plot_mean(results, "max_cit_count", "Macro-F1") +
    scale_x_log10(labels=comma) +
    theme_bw() +
    labs(x="Max Cit Count", y="Macro-F1", title="Citation Class Prediction") +
    theme(text=element_text(size=15)) +
    geom_text(aes(label=mean), nudge_x=0.15)
ggsave("plots/class_vary_cit_count.png", width=8, height=5)

targets = c("recall_1", "recall_5", "recall_20")
results = pretty_table("results/cit_idx_vary_cit_count.txt", 
                       split_cols=c("max_cit_count"))
plot_mean(results, "max_cit_count", targets) +
    scale_x_log10(labels=comma) +
    theme_bw() +
    labs(x="Max Cit Count", y="Recall", title="Full Citation Prediction") +
    theme(text=element_text(size=15)) +
    facet_grid(factor(variable, levels=targets)~., scales = "free_y") +
    geom_text(aes(label=mean), nudge_x=0.1) +
    theme(legend.position = "none")
ggsave("plots/class_vary_cit_idx.png", width=8, height=6)

# Letor Expts
results = pretty_table("results/cit_class_letor.txt",
                       split_cols=c("model", "features"),
                       labels=c("_mean", "_se", "_pval"))
print_overleaf(results)

results = pretty_table("results/cit_idx_letor.txt",
                       split_cols=c("model", "features"),
                       labels=c("_mean", "_se", "_pval"))
print_overleaf(results)

