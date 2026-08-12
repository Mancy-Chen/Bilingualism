###############################################################################
# BrainAge follow-up: Age–BAG analyses
# Uses reviewer-friendly BAG_raw / BAG_corr variable names.
###############################################################################

library(readxl)
library(dplyr)
library(ggplot2)
library(broom)
library(emmeans)
library(patchwork)

input_file <- file.path("input", "brainpad_results_deidentified.xlsx")
output_dir <- "output"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

df <- read_excel(input_file, sheet = "Analysis_Data") |>
  mutate(
    group = factor(
      group,
      levels = c("bilinguals", "translators", "interpreters")
    ),
    Gender = factor(Gender, levels = c("Female", "Male"))
  )

required <- c(
  "Age", "group", "Gender",
  "BAG_raw_BrainAge", "BAG_corr_BrainAge"
)
missing_cols <- setdiff(required, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

# ---------------------------------------------------------------------------
# Group-specific age slopes
# ---------------------------------------------------------------------------
raw_slopes <- df |>
  group_by(group) |>
  group_modify(~ tidy(lm(BAG_raw_BrainAge ~ Age, data = .x))) |>
  filter(term == "Age") |>
  ungroup()

corr_slopes <- df |>
  group_by(group) |>
  group_modify(~ tidy(lm(BAG_corr_BrainAge ~ Age, data = .x))) |>
  filter(term == "Age") |>
  ungroup()

write.csv(
  raw_slopes,
  file.path(output_dir, "age_slopes_BAG_raw_BrainAge.csv"),
  row.names = FALSE
)
write.csv(
  corr_slopes,
  file.path(output_dir, "age_slopes_BAG_corr_BrainAge.csv"),
  row.names = FALSE
)

print(raw_slopes)
print(corr_slopes)

# ---------------------------------------------------------------------------
# Combined Age × Group interaction models
# ---------------------------------------------------------------------------
lm_raw <- lm(BAG_raw_BrainAge ~ Age * group, data = df)
lm_corr <- lm(BAG_corr_BrainAge ~ Age * group, data = df)

print(summary(lm_raw))
print(anova(lm_raw))
print(summary(lm_corr))
print(anova(lm_corr))

write.csv(
  tidy(lm_raw),
  file.path(output_dir, "interaction_BAG_raw_BrainAge.csv"),
  row.names = FALSE
)
write.csv(
  tidy(lm_corr),
  file.path(output_dir, "interaction_BAG_corr_BrainAge.csv"),
  row.names = FALSE
)

# Pairwise comparison of Age slopes.
raw_emtrends <- emtrends(lm_raw, ~ group, var = "Age")
corr_emtrends <- emtrends(lm_corr, ~ group, var = "Age")

print(raw_emtrends)
print(pairs(raw_emtrends))
print(corr_emtrends)
print(pairs(corr_emtrends))

# ---------------------------------------------------------------------------
# Main age–BAG figure
# ---------------------------------------------------------------------------
plot_df <- df |>
  filter(
    !is.na(Age),
    !is.na(group),
    !is.na(BAG_raw_BrainAge),
    !is.na(BAG_corr_BrainAge)
  )

my_cols <- c(
  bilinguals = "#1f77b4",
  translators = "#ff7f0e",
  interpreters = "#2ca02c"
)

common_theme <- theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "right"
  )

p_raw <- ggplot(
  plot_df,
  aes(x = Age, y = BAG_raw_BrainAge, color = group, fill = group)
) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_smooth(method = "lm", se = TRUE, level = 0.95, alpha = 0.20) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_color_manual(values = my_cols, name = "Group") +
  scale_fill_manual(values = my_cols, guide = "none") +
  labs(
    x = "Age (years)",
    y = "BAG_raw (years)",
    title = "Age and uncorrected BAG"
  ) +
  common_theme

p_corr <- ggplot(
  plot_df,
  aes(x = Age, y = BAG_corr_BrainAge, color = group, fill = group)
) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_smooth(method = "lm", se = TRUE, level = 0.95, alpha = 0.20) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_color_manual(values = my_cols, name = "Group") +
  scale_fill_manual(values = my_cols, guide = "none") +
  labs(
    x = "Age (years)",
    y = "BAG_corr (years)",
    title = "Age and age-bias-corrected BAG"
  ) +
  common_theme

combined <- (p_raw + p_corr) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "right")

ggsave(
  file.path(output_dir, "Age_BAG_raw_BAG_corr_BrainAge.png"),
  combined,
  width = 12, height = 6, dpi = 600, bg = "white"
)

# ---------------------------------------------------------------------------
# Sex sensitivity: Age × Gender for BAG_corr
# ---------------------------------------------------------------------------
gender_df <- df |>
  filter(!is.na(Age), !is.na(Gender), !is.na(BAG_corr_BrainAge))

lm_gender <- lm(BAG_corr_BrainAge ~ Age * Gender, data = gender_df)
print(summary(lm_gender))
print(anova(lm_gender))

gender_slopes <- emtrends(lm_gender, ~ Gender, var = "Age")
print(gender_slopes)
print(pairs(gender_slopes))

p_gender <- ggplot(
  gender_df,
  aes(x = Age, y = BAG_corr_BrainAge, color = Gender, fill = Gender)
) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_smooth(method = "lm", se = TRUE, level = 0.95, alpha = 0.20) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    x = "Age (years)",
    y = "BAG_corr (years)",
    title = "Age–BAG_corr relationship by sex"
  ) +
  theme_minimal(base_size = 16)

ggsave(
  file.path(output_dir, "Gender_Age_BAG_corr_BrainAge.png"),
  p_gender,
  width = 8, height = 6, dpi = 600, bg = "white"
)
