###############################################################################
# Six-model Age × Group analysis and BrainAge sensitivity analyses
###############################################################################

library(readxl)
library(dplyr)
library(broom)

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

model_names <- c(
  "BrainAge",
  "BrainAgeR",
  "DeepBrainNet",
  "Pyment",
  "BRAID_WM",
  "BRAID_GM"
)

required <- c("Age", "group", paste0("BAG_corr_", model_names))
missing_cols <- setdiff(required, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

joint_interaction_test <- function(data, outcome) {
  dat <- data |>
    filter(!is.na(.data[[outcome]]), !is.na(Age), !is.na(group))

  reduced <- lm(
    as.formula(paste0("`", outcome, "` ~ Age + group")),
    data = dat
  )
  full <- lm(
    as.formula(paste0("`", outcome, "` ~ Age * group")),
    data = dat
  )
  cmp <- anova(reduced, full)

  data.frame(
    n = nobs(full),
    F = unname(cmp$F[2]),
    df_num = unname(cmp$Df[2]),
    df_den = df.residual(full),
    p = unname(cmp$`Pr(>F)`[2])
  )
}

# ---------------------------------------------------------------------------
# Six-model Age × Group interaction tests
# ---------------------------------------------------------------------------
interaction_rows <- lapply(model_names, function(model) {
  outcome <- paste0("BAG_corr_", model)
  out <- joint_interaction_test(df, outcome)
  out$model <- model
  out
})

interaction_summary <- bind_rows(interaction_rows) |>
  select(model, everything()) |>
  mutate(
    p_FDR = p.adjust(p, method = "BH"),
    FDR_significant = p_FDR < 0.05
  )

write.csv(
  interaction_summary,
  file.path(output_dir, "age_group_interactions_all_models.csv"),
  row.names = FALSE
)
print(interaction_summary)

# ---------------------------------------------------------------------------
# Reviewer-requested BrainAge sensitivity analyses
# ---------------------------------------------------------------------------
brainage_complete <- df |>
  filter(!is.na(BAG_corr_BrainAge), !is.na(Age), !is.na(group))

# Common age range across all three groups.
age_ranges <- brainage_complete |>
  group_by(group) |>
  summarise(
    min_age = min(Age),
    max_age = max(Age),
    .groups = "drop"
  )

common_min <- max(age_ranges$min_age)
common_max <- min(age_ranges$max_age)

common_age_df <- brainage_complete |>
  filter(Age >= common_min, Age <= common_max)

common_age_test <- joint_interaction_test(
  common_age_df,
  "BAG_corr_BrainAge"
)

# Cook's distance for the overall BrainAge Age × Group model.
brainage_full <- lm(
  BAG_corr_BrainAge ~ Age * group,
  data = brainage_complete
)
brainage_cook <- cooks.distance(brainage_full)
brainage_cook_threshold <- 4 / nobs(brainage_full)
brainage_flagged <- brainage_cook > brainage_cook_threshold

cook_excluded_df <- brainage_complete[!brainage_flagged, , drop = FALSE]
cook_test <- joint_interaction_test(
  cook_excluded_df,
  "BAG_corr_BrainAge"
)

# Interpreter-specific slope influence and robust regression.
interp_df <- brainage_complete |>
  filter(group == "interpreters")

interp_lm <- lm(BAG_corr_BrainAge ~ Age, data = interp_df)
interp_cook <- cooks.distance(interp_lm)
interp_cook_threshold <- 4 / nobs(interp_lm)
interp_flagged <- interp_cook > interp_cook_threshold

interp_excluded <- interp_df[!interp_flagged, , drop = FALSE]
interp_lm_excluded <- lm(BAG_corr_BrainAge ~ Age, data = interp_excluded)

orig_coef <- tidy(interp_lm) |>
  filter(term == "Age")
excl_coef <- tidy(interp_lm_excluded) |>
  filter(term == "Age")

if (!requireNamespace("MASS", quietly = TRUE)) {
  stop("The MASS package is required for Huber robust regression.")
}
interp_huber <- MASS::rlm(BAG_corr_BrainAge ~ Age, data = interp_df)
huber_summary <- summary(interp_huber)$coefficients
huber_beta <- unname(huber_summary["Age", "Value"])
huber_se <- unname(huber_summary["Age", "Std. Error"])
huber_z <- huber_beta / huber_se
huber_p <- 2 * pnorm(abs(huber_z), lower.tail = FALSE)

sensitivity_summary <- bind_rows(
  data.frame(
    analysis = "Common age range Age x Group interaction",
    n = common_age_test$n,
    common_age_min = common_min,
    common_age_max = common_max,
    n_flagged = NA_integer_,
    estimate = NA_real_,
    SE = NA_real_,
    F = common_age_test$F,
    df_num = common_age_test$df_num,
    df_den = common_age_test$df_den,
    p = common_age_test$p
  ),
  data.frame(
    analysis = "Cook-excluded Age x Group interaction",
    n = cook_test$n,
    common_age_min = NA_real_,
    common_age_max = NA_real_,
    n_flagged = sum(brainage_flagged),
    estimate = NA_real_,
    SE = NA_real_,
    F = cook_test$F,
    df_num = cook_test$df_num,
    df_den = cook_test$df_den,
    p = cook_test$p
  ),
  data.frame(
    analysis = "Interpreter original Age slope",
    n = nobs(interp_lm),
    common_age_min = NA_real_,
    common_age_max = NA_real_,
    n_flagged = sum(interp_flagged),
    estimate = orig_coef$estimate,
    SE = orig_coef$std.error,
    F = NA_real_,
    df_num = NA_real_,
    df_den = df.residual(interp_lm),
    p = orig_coef$p.value
  ),
  data.frame(
    analysis = "Interpreter Cook-excluded Age slope",
    n = nobs(interp_lm_excluded),
    common_age_min = NA_real_,
    common_age_max = NA_real_,
    n_flagged = sum(interp_flagged),
    estimate = excl_coef$estimate,
    SE = excl_coef$std.error,
    F = NA_real_,
    df_num = NA_real_,
    df_den = df.residual(interp_lm_excluded),
    p = excl_coef$p.value
  ),
  data.frame(
    analysis = "Interpreter Huber robust Age slope",
    n = nrow(interp_df),
    common_age_min = NA_real_,
    common_age_max = NA_real_,
    n_flagged = NA_integer_,
    estimate = huber_beta,
    SE = huber_se,
    F = NA_real_,
    df_num = NA_real_,
    df_den = NA_real_,
    p = huber_p
  )
)

write.csv(
  age_ranges,
  file.path(output_dir, "age_ranges_by_group.csv"),
  row.names = FALSE
)
write.csv(
  sensitivity_summary,
  file.path(output_dir, "BrainAge_sensitivity_analyses.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    row_index = which(brainage_flagged),
    cooks_distance = brainage_cook[brainage_flagged],
    threshold = brainage_cook_threshold
  ),
  file.path(output_dir, "BrainAge_cooks_distance_flagged.csv"),
  row.names = FALSE
)

print(sensitivity_summary)
