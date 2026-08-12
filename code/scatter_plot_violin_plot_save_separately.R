###############################################################################
# Manuscript-style BrainAge violin + scatter figures
# Uses BAG_raw_BrainAge and BAG_corr_BrainAge.
###############################################################################

library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(rstatix)
library(ggpubr)
library(patchwork)

input_file <- file.path("input", "brainpad_results_deidentified.xlsx")
output_dir <- "output"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

df <- read_excel(input_file, sheet = "Analysis_Data") |>
  mutate(
    group = factor(
      group,
      levels = c("bilinguals", "translators", "interpreters")
    )
  )

df_long <- df |>
  select(group, Age, BAG_raw_BrainAge, BAG_corr_BrainAge) |>
  pivot_longer(
    c(BAG_raw_BrainAge, BAG_corr_BrainAge),
    names_to = "type",
    values_to = "BAG"
  ) |>
  mutate(
    type = recode(
      type,
      BAG_raw_BrainAge = "BAG_raw",
      BAG_corr_BrainAge = "BAG_corr"
    )
  )

my_cols <- c(
  bilinguals = "#1f77b4",
  translators = "#ff7f0e",
  interpreters = "#2ca02c"
)

make_pair <- function(type_name, title_text, output_name) {
  dat <- df_long |> filter(type == type_name)

  omnibus <- anova_test(dat, BAG ~ group)
  print(omnibus)

  tukey <- dat |>
    tukey_hsd(BAG ~ group) |>
    filter(p.adj < 0.05)

  if (nrow(tukey) > 0) {
    tukey <- tukey |>
      add_xy_position(x = "group", fun = "max") |>
      mutate(
        p.label = ifelse(
          p.adj < 0.001,
          "p < 0.001",
          sprintf("p = %.3f", p.adj)
        )
      )
  }

  p_violin <- ggplot(dat, aes(x = group, y = BAG, fill = group)) +
    geom_violin(trim = FALSE, alpha = 0.7, linewidth = 1.1) +
    geom_boxplot(
      width = 0.15,
      outlier.shape = NA,
      color = "black",
      linewidth = 0.9
    ) +
    stat_summary(
      fun = mean, geom = "point",
      shape = 21, size = 3.2,
      fill = "white", color = "black"
    ) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_fill_manual(values = my_cols, guide = "none") +
    labs(
      x = "Group",
      y = paste0(type_name, " (years)"),
      title = paste("Group distribution of", type_name)
    ) +
    theme_minimal(base_size = 16)

  if (nrow(tukey) > 0) {
    p_violin <- p_violin +
      stat_pvalue_manual(
        tukey,
        label = "p.label",
        tip.length = 0.01,
        bracket.size = 0.7,
        size = 4.5
      )
  }

  wide_col <- if (type_name == "BAG_raw") {
    "BAG_raw_BrainAge"
  } else {
    "BAG_corr_BrainAge"
  }

  scatter_dat <- df |>
    filter(!is.na(Age), !is.na(group), !is.na(.data[[wide_col]]))

  p_scatter <- ggplot(
    scatter_dat,
    aes(
      x = Age,
      y = .data[[wide_col]],
      color = group,
      fill = group
    )
  ) +
    geom_point(alpha = 0.7, size = 2.5) +
    geom_smooth(method = "lm", se = TRUE, level = 0.95, alpha = 0.20) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_color_manual(values = my_cols, name = "Group") +
    scale_fill_manual(values = my_cols, guide = "none") +
    labs(
      x = "Age (years)",
      y = paste0(type_name, " (years)"),
      title = title_text
    ) +
    theme_minimal(base_size = 16)

  combined <- (p_violin + p_scatter) +
    plot_layout(ncol = 2, guides = "collect") &
    theme(legend.position = "right")

  ggsave(
    file.path(output_dir, output_name),
    combined,
    width = 14, height = 7, dpi = 600, bg = "white"
  )
}

make_pair(
  "BAG_raw",
  "Linear association between age and BAG_raw",
  "Figure_BAG_raw_BrainAge.png"
)

make_pair(
  "BAG_corr",
  "Linear association between age and BAG_corr",
  "Figure_BAG_corr_BrainAge.png"
)
