# Load data
# Install if needed:
# install.packages(c("readxl", "dplyr", "ggplot2", "broom"))

library(readxl)
library(dplyr)
library(ggplot2)
library(broom)

# Read your Excel file (note the forward slashes)
df <- read_excel("C:/brainpad_results.xlsx")

# Make sure the grouping variable is a factor and ordered properly
df <- df |>
  mutate(
    group = factor(group,
                   levels = c("bilinguals", "translators", "interpreters"))
  )
############################################################################
# Scatter plot: Predicted_BAG_non_BC_Brainage vs Age
p1 <- ggplot(df, aes(x = Age,
                     y = Predicted_BAG_non_BC_Brainage,
                     color = group)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  scale_color_manual(values = c(bilinguals="#1f77b4", translators="#ff7f0e", interpreters="#2ca02c"
  )) +
  labs(
    x = "Age",
    y = "Predicted BAG (non-BC Brainage)",
    title = "Predicted BAG vs Age by Group"
  ) +
  theme_minimal()

print(p1)

# Optional: save to file
#ggsave("Predicted_BAG_vs_Age_by_Group.png", p1, width = 7, height = 5, dpi = 300)

# Scatter plot: delta_cv5_Predicted_age_non_BC_Brainage vs Age
p2 <- ggplot(df, aes(x = Age,
                     y = delta_cv5_Predicted_age_non_BC_Brainage,
                     color = group)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  scale_color_manual(values = c(
    bilinguals="#1f77b4", translators="#ff7f0e", interpreters="#2ca02c"
  )) +
  labs(
    x = "Age",
    y = "Δ cv5 Predicted Age (non-BC Brainage)",
    title = "Δ Predicted Age vs Age by Group"
  ) +
  theme_minimal()

print(p2)

# Optional:
#ggsave("Delta_cv5_Predicted_Age_vs_Age_by_Group.png", p2, width = 7, height = 5, dpi = 300)


################################################################################
# Statistical tests of the slope
# raw BAG
bag_age_models <- df |>
  group_by(group) |>
  do(tidy(lm(Predicted_BAG_non_BC_Brainage ~ Age, data = .))) |>
  ungroup() |>
  filter(term == "Age")

bag_age_models
# CV5 BAG
delta_age_models <- df |>
  group_by(group) |>
  do(tidy(lm(delta_cv5_Predicted_age_non_BC_Brainage ~ Age, data = .))) |>
  ungroup() |>
  filter(term == "Age")

delta_age_models
################################################################################
# Optional:single model with Group × Age interaction
# Model for BAG
lm_bag <- lm(Predicted_BAG_non_BC_Brainage ~ Age * group, data = df)
summary(lm_bag)
anova(lm_bag)  # tests interaction etc.

# Model for delta
lm_delta <- lm(delta_cv5_Predicted_age_non_BC_Brainage ~ Age * group, data = df)
summary(lm_delta)
anova(lm_delta)

###############################################################################
# Comparison among slopes:
# raw BAG
# install.packages("emmeans")  # if not installed
library(emmeans)

# Get estimated Age slopes per group
bag_slopes <- emtrends(lm_bag, ~ group, var = "Age")
bag_slopes          # shows slope for Age in each group

# Pairwise comparisons of slopes:
pairs(bag_slopes)   # bilinguals vs translators, translators vs interpreters, bilinguals vs interpreters

# Corrected BAG
lm_delta <- lm(delta_cv5_Predicted_age_non_BC_Brainage ~ Age * group, data = df)

delta_slopes <- emtrends(lm_delta, ~ group, var = "Age")
delta_slopes

pairs(delta_slopes)
###############################################################################
# Scatter plot with two subplots (raw/corrected BAG)
library(ggplot2)
library(emmeans)
library(patchwork)

# Models
lm_bag   <- lm(Predicted_BAG_non_BC_Brainage ~ Age * group, data = df)
lm_delta <- lm(delta_cv5_Predicted_age_non_BC_Brainage ~ Age * group, data = df)

# Plot 1: Raw BAG
p_bag <- ggplot(df, aes(x = Age,
                        y = Predicted_BAG_non_BC_Brainage,
                        color = group)) +
  geom_point(alpha = 0.7, size = 3) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.4) +
  scale_color_manual(
    values = c(
      bilinguals    = "#1f77b4",
      translators   = "#ff7f0e",
      interpreters  = "#2ca02c"
    ),
    name = "Group",
    labels = c("Bilinguals", "Translators", "Interpreters")
  ) +
  scale_y_continuous(
    breaks = seq(-20, 20, 10),   # -20, -10, 0, 10, 20
    limits = c(-20, 20)
  ) +
  labs(
    x = "Age (years)",
    y = "Raw BAG",
    title = "Raw BAG by age"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", size = 16),
    axis.title = element_text(size = 16),
    axis.text  = element_text(size = 14),
    legend.title = element_text(size = 14),
    legend.text  = element_text(size = 13)
  )

# Plot 2: Corrected BAG
p_delta <- ggplot(df, aes(x = Age,
                          y = delta_cv5_Predicted_age_non_BC_Brainage,
                          color = group)) +
  geom_point(alpha = 0.7, size = 3) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.4) +
  scale_color_manual(
    values = c(
      bilinguals    = "#1f77b4",
      translators   = "#ff7f0e",
      interpreters  = "#2ca02c"
    ),
    name = "Group",
    labels = c("Bilinguals", "Translators", "Interpreters")
  ) +
  scale_y_continuous(
    breaks = seq(-20, 20, 10),
    limits = c(-20, 20)
  ) +
  labs(
    x = "Age (years)",
    y = "Corrected BAG",
    title = "Corrected BAG by age"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", size = 16),
    axis.title = element_text(size = 16),
    axis.text  = element_text(size = 14),
    legend.title = element_text(size = 14),
    legend.text  = element_text(size = 13)
  )

# Combine horizontally, shared legend
combined_plot <- p_bag + p_delta +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "right")

print(combined_plot)

ggsave("Fig_BAG_horizontal_symm_scale.png",
       combined_plot,
       width = 10, height = 5, dpi = 300)
###############################################################################
###############################################################################
# GRAPH 1: BAG violins (Uncorrected left, Corrected right) + ANOVA/Tukey
# Brackets ONLY for p.adj < 0.05, lifted above violins
###############################################################################

library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(rstatix)
library(ggpubr)

df <- read_excel("C:/brainpad_results.xlsx") %>%
  mutate(group = factor(group, levels = c("bilinguals","translators","interpreters"))) %>%
  mutate(
    BAG_uncorrected = Predicted_BAG_non_BC_Brainage,
    BAG_corrected   = delta_cv5_Predicted_age_non_BC_Brainage
  )

my_cols <- c(
  bilinguals   = "#1f77b4",
  translators  = "#ff7f0e",
  interpreters = "#2ca02c"
)

df_long <- df %>%
  pivot_longer(c(BAG_uncorrected, BAG_corrected),
               names_to = "type", values_to = "BAG") %>%
  mutate(
    type = recode(type,
                  BAG_uncorrected = "Uncorrected BAG",
                  BAG_corrected   = "Corrected BAG"),
    type = factor(type, levels = c("Uncorrected BAG", "Corrected BAG"))
  )

# ---- ANOVA per panel (prints) ----
anova_bag <- df_long %>% group_by(type) %>% anova_test(BAG ~ group)
print(anova_bag)

# ---- Compute BAG range per panel (needed to lift brackets) ----
bag_range <- df_long %>%
  group_by(type) %>%
  summarise(
    rng = max(BAG, na.rm = TRUE) - min(BAG, na.rm = TRUE),
    .groups = "drop"
  )

# ---- Tukey (significant only) + bracket positions ----
step_inc  <- 0.18
lift_frac <- 0.15

tukey_bag_sig <- df_long %>%
  group_by(type) %>%
  tukey_hsd(BAG ~ group) %>%
  ungroup() %>%
  filter(p.adj < 0.05) %>%                         # ONLY significant
  left_join(bag_range, by = "type") %>%            # add rng per panel
  add_xy_position(x = "group", fun = "max", step.increase = step_inc) %>%
  mutate(
    y.position = y.position + lift_frac * rng,
    p.label = p_format(p.adj, digits = 3)
  )

p_bag <- ggplot(df_long, aes(x = group, y = BAG, fill = group)) +
  geom_violin(trim = FALSE, alpha = 0.7, linewidth = 1.3) +
  geom_boxplot(width = 0.15, outlier.shape = NA, color = "black", linewidth = 1.1) +
  stat_summary(fun = mean, geom = "point", shape = 21, size = 3.5,
               stroke = 1.0, fill = "white", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1.2) +
  scale_fill_manual(values = my_cols, name = "Group",
                    labels = c("Bilinguals","Translators","Interpreters")) +
  facet_wrap(~type, nrow = 1) +
  { if (nrow(tukey_bag_sig) > 0)
    stat_pvalue_manual(tukey_bag_sig, label = "p.label",
                       tip.length = 0.01, bracket.size = 0.8, size = 6)
    else NULL
  } +
  labs(x = "Group", y = "BAG (years)",
       title = "Brainage BAG distributions (all ages) + ANOVA/Tukey (p<0.05 only)") +
  theme_minimal(base_size = 20) +
  theme(
    plot.title = element_text(face = "bold", size = 22),
    strip.text = element_text(face = "bold", size = 20),
    axis.title = element_text(size = 20),
    axis.text  = element_text(size = 18),
    legend.position = "right",
    legend.title = element_text(size = 18, face = "bold"),
    legend.text  = element_text(size = 16)
  )

print(p_bag)

# Optional save
# ggsave("Violin_BAG_uncorrected_left_corrected_right_ANOVA_Tukey_sigOnly.png",
#        p_bag, width = 13, height = 5, dpi = 300)



###############################################################################
# GRAPH 2: Age-only violin + ANOVA/Tukey
# Brackets ONLY for p.adj < 0.05, lifted above violins
###############################################################################

library(readxl)
library(dplyr)
library(ggplot2)
library(rstatix)
library(ggpubr)

df <- read_excel("C:/brainpad_results.xlsx") %>%
  mutate(group = factor(group, levels = c("bilinguals","translators","interpreters")))

my_cols <- c(
  bilinguals   = "#1f77b4",
  translators  = "#ff7f0e",
  interpreters = "#2ca02c"
)

# ANOVA (prints)
anova_age <- anova_test(df, Age ~ group)
print(anova_age)

# Tukey (significant only) + bracket positions
step_inc  <- 0.18
lift_frac <- 0.40
age_rng   <- max(df$Age, na.rm = TRUE) - min(df$Age, na.rm = TRUE)

tukey_age_sig <- tukey_hsd(df, Age ~ group) %>%
  filter(p.adj < 0.05) %>%
  add_xy_position(x = "group", fun = "max", step.increase = step_inc) %>%
  mutate(
    y.position = y.position + lift_frac * age_rng,
    p.label = p_format(p.adj, digits = 3)
  )

p_age <- ggplot(df, aes(x = group, y = Age, fill = group)) +
  geom_violin(trim = FALSE, alpha = 0.7, linewidth = 1.3) +
  geom_boxplot(width = 0.15, outlier.shape = NA, color = "black", linewidth = 1.1) +
  stat_summary(fun = mean, geom = "point", shape = 21, size = 3.5,
               stroke = 1.0, fill = "white", color = "black") +
  scale_fill_manual(values = my_cols, name = "Group",
                    labels = c("Bilinguals","Translators","Interpreters")) +
  { if (nrow(tukey_age_sig) > 0)
    stat_pvalue_manual(tukey_age_sig, label = "p.label",
                       tip.length = 0.01, bracket.size = 0.8, size = 6)
    else NULL
  } +
  labs(x = "Group", y = "Age (years)",
       title = "Age distributions by group + ANOVA/Tukey (p<0.05 only)") +
  theme_minimal(base_size = 20) +
  theme(
    plot.title = element_text(face = "bold", size = 22),
    axis.title = element_text(size = 20),
    axis.text  = element_text(size = 18),
    legend.position = "right",
    legend.title = element_text(size = 18, face = "bold"),
    legend.text  = element_text(size = 16)
  )

print(p_age)

# Optional save
# ggsave("Violin_Age_by_group_ANOVA_Tukey_sigOnly.png",
#        p_age, width = 7, height = 5, dpi = 300)


###############################################################################
# Chronological age vs predicted age
# Chronological age vs predicted age (two panels with identical x/y scales)

library(ggplot2)
library(patchwork)
library(dplyr)
library(readxl)

# Read data (edit path if needed)
df <- read_excel("C:/brainpad_results.xlsx")

# Ensure group is ordered
df <- df |>
  mutate(group = factor(group, levels = c("bilinguals", "translators", "interpreters")))

# Create predicted age variables (brain age)
df <- df |>
  mutate(
    Predicted_Age_uncorrected = Age + Predicted_BAG_non_BC_Brainage,
    Predicted_Age_corrected   = Age + delta_cv5_Predicted_age_non_BC_Brainage
  )

# Shared axis limits across both panels
x_lim <- range(df$Age, na.rm = TRUE)
y_lim <- range(
  c(df$Predicted_Age_uncorrected, df$Predicted_Age_corrected),
  na.rm = TRUE
)

# Colors
my_cols <- c(
  bilinguals = "#1f77b4",
  translators = "#ff7f0e",
  interpreters = "#2ca02c"
)

# Uncorrected
p3 <- ggplot(df, aes(x = Age, y = Predicted_Age_uncorrected, color = group)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 1.4) +
  scale_color_manual(values = my_cols) +
  scale_x_continuous(limits = x_lim) +
  scale_y_continuous(limits = y_lim) +
  coord_fixed(ratio = 1) +  # same units on x and y for both panels
  labs(
    x = "Chronological Age",
    y = "Predicted Brain Age (uncorrected)",
    title = "Uncorrected",
    color = "Group"
  ) +
  theme_minimal()

# Bias-corrected
p4 <- ggplot(df, aes(x = Age, y = Predicted_Age_corrected, color = group)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 1.4) +
  scale_color_manual(values = my_cols) +
  scale_x_continuous(limits = x_lim) +
  scale_y_continuous(limits = y_lim) +
  coord_fixed(ratio = 1) +
  labs(
    x = "Chronological Age",
    y = "Predicted Brain Age (bias-corrected)",
    title = "Bias-corrected",
    color = "Group"
  ) +
  theme_minimal()

# Combine with one shared legend on the right
combined <- (p3 + p4) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "right")

print(combined)

# Save
ggsave(
  "ChronAge_vs_PredAge_combined_shared_legend_right.png",
  combined, width = 12, height = 5, dpi = 300
)


########################################################################################
# Gender difference
library(readxl)
library(dplyr)
library(ggplot2)
library(dplyr)
library(readxl)
library(rstatix)
library(ggpubr)

df <- read_excel("C:/brainpad_results.xlsx") |>
  mutate(
    group  = factor(group,
                    levels = c("bilinguals", "translators", "interpreters")),
    Gender = factor(Gender, levels = c("Female", "Male"))
  )

# Violin + boxplot
p_bag_violin <- ggplot(
  df,
  aes(x = group,
      y = delta_cv5_Predicted_age_non_BC_Brainage,
      fill = Gender)
) +
  geom_violin(trim = FALSE,
              position = position_dodge(width = 0.9),
              alpha = 0.7) +
  geom_boxplot(width = 0.15,
               position = position_dodge(width = 0.9),
               outlier.shape = NA,
               color = "black") +
  stat_summary(fun = mean,
               geom = "point",
               position = position_dodge(width = 0.9),
               shape = 21, size = 2.5,
               fill = "white", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_fill_manual(
    values = c("Female" = "#F8766D",
               "Male"   = "#00BFC4"),
    name = "Gender"
  ) +
  labs(
    x = "Group",
    y = "Corrected BAG (years)",
    title = "Corrected BAG by Group and Gender"
  ) +
  theme_minimal(base_size = 16)

# ---- t-tests Female vs Male within each group ----
library(rstatix)
library(ggpubr)

# t-tests Female vs Male in each group, then correct across groups
stat.test <- df %>%
  group_by(group) %>%
  t_test(delta_cv5_Predicted_age_non_BC_Brainage ~ Gender) %>%
  ungroup() %>%
  mutate(
    p.adj   = p.adjust(p, method = "holm"),   # or "bonferroni", "fdr", etc.
    p.label = rstatix::p_format(p.adj, digits = 3)
  ) %>%
  add_xy_position(x = "group")

p_bag_violin +
  stat_pvalue_manual(
    stat.test,
    label = "p.label",
    tip.length = 0.01,
    size = 4
  )

################################################################################
# Gender difference for whole group
library(emmeans)
library(ggplot2)
library(dplyr)

df <- read_excel("C:/brainpad_results.xlsx") |>
  mutate(
    Gender = factor(Gender, levels = c("Female", "Male"))
  )

p_gender_age <- ggplot(
  df,
  aes(x = Age,
      y = delta_cv5_Predicted_age_non_BC_Brainage,
      color = Gender)
) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_color_manual(
    values = c("Female" = "#F8766D",
               "Male"   = "#00BFC4"),
    name = "Gender"
  ) +
  labs(
    x = "Age (years)",
    y = "Corrected BAG (years)",
    title = "Age–corrected BAG relationship by gender (all groups combined)"
  ) +
  theme_minimal(base_size = 16)

print(p_gender_age)

# OLS model with Age × Gender interaction (all groups pooled)
lm_gender <- lm(
  delta_cv5_Predicted_age_non_BC_Brainage ~ Age * Gender,
  data = df
)

summary(lm_gender)
anova(lm_gender)   # tests main effects + Age:Gender interaction

# Estimated age slopes per gender
gender_slopes <- emtrends(lm_gender, ~ Gender, var = "Age")
gender_slopes

# Pairwise comparison of slopes (Female vs Male)
pairs(gender_slopes)

