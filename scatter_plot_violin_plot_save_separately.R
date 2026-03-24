###############################################################################
# COMBINED FIGURES SAVED SEPARATELY
# 1) Uncorrected BAG: violin (left, A) + scatter (right, B)
# 2) Corrected BAG:   violin (left, A) + scatter (right, B)
# Shared legend on the right
###############################################################################

library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(rstatix)
library(ggpubr)
library(patchwork)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df <- read_excel("E:/Bilingualism/brainpad_results.xlsx") %>%
  mutate(
    group = factor(group, levels = c("bilinguals", "translators", "interpreters")),
    BAG_uncorrected = Predicted_BAG_non_BC_Brainage,
    BAG_corrected   = delta_cv5_Predicted_age_non_BC_Brainage
  )

# Colors
my_cols <- c(
  bilinguals   = "#1f77b4",
  translators  = "#ff7f0e",
  interpreters = "#2ca02c"
)

# -----------------------------------------------------------------------------
# Long format for violin plots
# -----------------------------------------------------------------------------
df_long <- df %>%
  pivot_longer(
    c(BAG_uncorrected, BAG_corrected),
    names_to = "type",
    values_to = "BAG"
  ) %>%
  mutate(
    type = recode(
      type,
      BAG_uncorrected = "Uncorrected BAG",
      BAG_corrected   = "Corrected BAG"
    ),
    type = factor(type, levels = c("Uncorrected BAG", "Corrected BAG"))
  )

# ANOVA
anova_bag <- df_long %>%
  group_by(type) %>%
  anova_test(BAG ~ group)

print(anova_bag)

# BAG range per panel
bag_range <- df_long %>%
  group_by(type) %>%
  summarise(
    rng = max(BAG, na.rm = TRUE) - min(BAG, na.rm = TRUE),
    .groups = "drop"
  )

# Significant Tukey only
step_inc  <- 0.18
lift_frac <- 0.15

tukey_bag_sig <- df_long %>%
  group_by(type) %>%
  tukey_hsd(BAG ~ group) %>%
  ungroup() %>%
  filter(p.adj < 0.05) %>%
  left_join(bag_range, by = "type") %>%
  add_xy_position(x = "group", fun = "max", step.increase = step_inc) %>%
  mutate(
    y.position = y.position + lift_frac * rng,
    p.label = p_format(p.adj, digits = 3)
  )

# -----------------------------------------------------------------------------
# Data for scatter plots
# -----------------------------------------------------------------------------
df_plot <- df %>%
  filter(
    !is.na(Age),
    !is.na(group),
    !is.na(BAG_uncorrected),
    !is.na(BAG_corrected)
  )

# Common y-axis range for scatter plots
y_min <- floor(min(c(df_plot$BAG_uncorrected, df_plot$BAG_corrected), na.rm = TRUE))
y_max <- ceiling(max(c(df_plot$BAG_uncorrected, df_plot$BAG_corrected), na.rm = TRUE))
abs_y <- max(abs(y_min), abs(y_max))
y_min <- -abs_y
y_max <-  abs_y
y_breaks <- pretty(c(y_min, y_max), n = 5)

# Common x-axis range for scatter plots
x_min <- floor(min(df_plot$Age, na.rm = TRUE))
x_max <- ceiling(max(df_plot$Age, na.rm = TRUE))

# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
big_theme <- theme_minimal(base_size = 20) +
  theme(
    plot.title   = element_text(face = "bold", size = 22, hjust = 0.5),
    axis.title   = element_text(size = 20),
    axis.text    = element_text(size = 18),
    legend.title = element_text(size = 18, face = "bold"),
    legend.text  = element_text(size = 16)
  )

# -----------------------------------------------------------------------------
# 1) UNCORRECTED VIOLIN (A)
# -----------------------------------------------------------------------------
df_unc <- df_long %>% filter(type == "Uncorrected BAG")
tukey_unc <- tukey_bag_sig %>% filter(type == "Uncorrected BAG")

p_unc_violin <- ggplot(df_unc, aes(x = group, y = BAG, fill = group)) +
  geom_violin(trim = FALSE, alpha = 0.7, linewidth = 1.3) +
  geom_boxplot(width = 0.15, outlier.shape = NA, color = "black", linewidth = 1.1) +
  stat_summary(
    fun = mean, geom = "point", shape = 21, size = 3.5,
    stroke = 1.0, fill = "white", color = "black"
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1.2) +
  scale_fill_manual(
    values = my_cols,
    name = "Group",
    labels = c("Bilinguals", "Translators", "Interpreters")
  ) +
  {
    if (nrow(tukey_unc) > 0)
      stat_pvalue_manual(
        tukey_unc, label = "p.label",
        tip.length = 0.01, bracket.size = 0.8, size = 6
      )
    else NULL
  } +
  labs(
    x = "Group",
    y = "BAG (years)",
    title = "Uncorrected BAG"
  ) +
  big_theme +
  theme(
    legend.position = "right"
  ) +
  annotate(
    "text",
    x = Inf, y = Inf, label = "A",
    hjust = 1.1, vjust = 1.2,
    size = 10, fontface = "bold"
  )

# -----------------------------------------------------------------------------
# 2) UNCORRECTED SCATTER (B)
# -----------------------------------------------------------------------------
p_unc_scatter <- ggplot(
  df_plot,
  aes(x = Age, y = BAG_uncorrected, color = group, fill = group)
) +
  geom_point(alpha = 0.7, size = 3) +
  geom_smooth(
    method = "lm", se = TRUE, level = 0.95,
    linewidth = 1.4, alpha = 0.20
  ) +
  scale_color_manual(
    values = my_cols,
    name = "Group",
    labels = c("Bilinguals", "Translators", "Interpreters")
  ) +
  scale_fill_manual(
    values = my_cols,
    name = "Group",
    labels = c("Bilinguals", "Translators", "Interpreters")
  ) +
  scale_x_continuous(limits = c(x_min, x_max)) +
  scale_y_continuous(limits = c(y_min, y_max), breaks = y_breaks) +
  coord_fixed(ratio = 1) +
  labs(
    x = "Age (years)",
    y = "Uncorrected BAG",
    title = "Uncorrected BAG by age"
  ) +
  big_theme +
  annotate(
    "text",
    x = Inf, y = Inf, label = "B",
    hjust = 1.1, vjust = 1.2,
    size = 10, fontface = "bold"
  )

# -----------------------------------------------------------------------------
# 3) CORRECTED VIOLIN (A)
# -----------------------------------------------------------------------------
df_cor <- df_long %>% filter(type == "Corrected BAG")
tukey_cor <- tukey_bag_sig %>% filter(type == "Corrected BAG")

p_cor_violin <- ggplot(df_cor, aes(x = group, y = BAG, fill = group)) +
  geom_violin(trim = FALSE, alpha = 0.7, linewidth = 1.3) +
  geom_boxplot(width = 0.15, outlier.shape = NA, color = "black", linewidth = 1.1) +
  stat_summary(
    fun = mean, geom = "point", shape = 21, size = 3.5,
    stroke = 1.0, fill = "white", color = "black"
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1.2) +
  scale_fill_manual(
    values = my_cols,
    name = "Group",
    labels = c("Bilinguals", "Translators", "Interpreters")
  ) +
  {
    if (nrow(tukey_cor) > 0)
      stat_pvalue_manual(
        tukey_cor, label = "p.label",
        tip.length = 0.01, bracket.size = 0.8, size = 6
      )
    else NULL
  } +
  labs(
    x = "Group",
    y = "BAG (years)",
    title = "Corrected BAG"
  ) +
  big_theme +
  theme(
    legend.position = "right"
  ) +
  annotate(
    "text",
    x = Inf, y = Inf, label = "A",
    hjust = 1.1, vjust = 1.2,
    size = 10, fontface = "bold"
  )

# -----------------------------------------------------------------------------
# 4) CORRECTED SCATTER (B)
# -----------------------------------------------------------------------------
p_cor_scatter <- ggplot(
  df_plot,
  aes(x = Age, y = BAG_corrected, color = group, fill = group)
) +
  geom_point(alpha = 0.7, size = 3) +
  geom_smooth(
    method = "lm", se = TRUE, level = 0.95,
    linewidth = 1.4, alpha = 0.20
  ) +
  scale_color_manual(
    values = my_cols,
    name = "Group",
    labels = c("Bilinguals", "Translators", "Interpreters")
  ) +
  scale_fill_manual(
    values = my_cols,
    name = "Group",
    labels = c("Bilinguals", "Translators", "Interpreters")
  ) +
  scale_x_continuous(limits = c(x_min, x_max)) +
  scale_y_continuous(limits = c(y_min, y_max), breaks = y_breaks) +
  coord_fixed(ratio = 1) +
  labs(
    x = "Age (years)",
    y = "Corrected BAG",
    title = "Corrected BAG by age"
  ) +
  big_theme +
  annotate(
    "text",
    x = Inf, y = Inf, label = "B",
    hjust = 1.1, vjust = 1.2,
    size = 10, fontface = "bold"
  )

# -----------------------------------------------------------------------------
# Combine: UNCORRECTED
# -----------------------------------------------------------------------------
fig_unc <- (p_unc_violin + p_unc_scatter) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "right")

print(fig_unc)

ggsave(
  "E:/Bilingualism/Fig_Uncorrected_violin_scatter_sharedLegend_600dpi.png",
  fig_unc,
  width = 14, height = 7, units = "in",
  dpi = 600, bg = "white"
)

# -----------------------------------------------------------------------------
# Combine: CORRECTED
# -----------------------------------------------------------------------------
fig_cor <- (p_cor_violin + p_cor_scatter) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "right")

print(fig_cor)

ggsave(
  "E:/Bilingualism/Fig_Corrected_violin_scatter_sharedLegend_600dpi.png",
  fig_cor,
  width = 14, height = 7, units = "in",
  dpi = 600, bg = "white"
)