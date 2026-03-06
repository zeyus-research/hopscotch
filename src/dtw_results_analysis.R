library(conflicted)
library(lme4)
library(lmerTest)
library(tidyverse)
library(patchwork)
library(emmeans)
library(report)      # For easystats reporting
library(performance) # For model diagnostics
library(see)         # For visualizations
library(parameters)  # For effect sizes
library(moments)  # for skewness()
conflicts_prefer(dplyr::filter)
conflicts_prefer(lme4::lmer)
conflicts_prefer(moments::skewness)

# Load in the DTW results
# dtw_results <- read_tsv("analysis/dtw_results_20250822_154121.tsv")
dtw_results <- read_tsv("hopscotch_results/dtw_joint_angles_results_20260123_151036.tsv")

dtw_results <- dtw_results %>%
  mutate(
    subject_1 = as.factor(subject_1),
    subject_2 = as.factor(subject_2),
    condition_1 = as.factor(condition_1),
    condition_2 = as.factor(condition_2),
    obstacles_1 = as.factor(obstacles_1),
    obstacles_2 = as.factor(obstacles_2),
    # Add log-transformed DTW distance
    log_dtw_distance = log(dtw_distance)
  )

within_data <- dtw_results %>%
  filter(dtw_type == "within") %>%
  mutate(
    condition_pair = paste(
      pmin(as.character(condition_1), as.character(condition_2)),
      pmax(as.character(condition_1), as.character(condition_2)),
      sep = "-"
    ),
    condition_pair = as.factor(condition_pair),
    obstacles_match = as.factor(obstacles_1 == obstacles_2)
  )

# ============================================================================
# DIAGNOSTIC: Should we use log transformation?
# ============================================================================

# Create a diagnostic directory
if (!dir.exists("analysis_results/diagnostics")) {
  dir.create("analysis_results/diagnostics", recursive = TRUE)
}

# 1. Check distribution of raw DTW distances
p_raw_hist <- ggplot(dtw_results, aes(x = dtw_distance)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
  labs(title = "Distribution of Raw DTW Distances",
       x = "DTW Distance", y = "Count") +
  theme_minimal()

p_log_hist <- ggplot(dtw_results, aes(x = log_dtw_distance)) +
  geom_histogram(bins = 50, fill = "coral", alpha = 0.7) +
  labs(title = "Distribution of Log-Transformed DTW Distances",
       x = "Log(DTW Distance)", y = "Count") +
  theme_minimal()


p_dist <- p_raw_hist / p_log_hist
print(p_dist)
ggsave("analysis_results/diagnostics/distribution_comparison.png", p_dist, 
       width = 10, height = 8, dpi = 300)

# 2. Q-Q plots
p_qq_raw <- ggplot(dtw_results, aes(sample = dtw_distance)) +
  stat_qq() + stat_qq_line() +
  labs(title = "Q-Q Plot: Raw DTW Distances") +
  theme_minimal()

p_qq_log <- ggplot(dtw_results, aes(sample = log_dtw_distance)) +
  stat_qq() + stat_qq_line() +
  labs(title = "Q-Q Plot: Log-Transformed DTW Distances") +
  theme_minimal()

p_qq <- p_qq_raw | p_qq_log
print(p_qq)
ggsave("analysis_results/diagnostics/qq_plots.png", p_qq, 
       width = 12, height = 5, dpi = 300)

# 3. Skewness comparison

cat("\n=== DISTRIBUTION STATISTICS ===\n")
cat("Raw DTW Distance:\n")
cat(sprintf("  Skewness: %.3f\n", skewness(dtw_results$dtw_distance)))
cat(sprintf("  Mean: %.2f, Median: %.2f, SD: %.2f\n", 
            mean(dtw_results$dtw_distance), 
            median(dtw_results$dtw_distance),
            sd(dtw_results$dtw_distance)))

cat("\nLog-Transformed DTW Distance:\n")
cat(sprintf("  Skewness: %.3f\n", skewness(dtw_results$log_dtw_distance)))
cat(sprintf("  Mean: %.2f, Median: %.2f, SD: %.2f\n", 
            mean(dtw_results$log_dtw_distance), 
            median(dtw_results$log_dtw_distance),
            sd(dtw_results$log_dtw_distance)))

# 4. Compare model residuals
# Fit models with raw data
within_matched_raw <- within_data %>% filter(obstacles_match == TRUE)

model_raw <- lmer(
  dtw_distance ~ condition_pair + (1 | subject_1),
  data = within_matched_raw,
  REML = FALSE
)

model_log <- lmer(
  log_dtw_distance ~ condition_pair + (1 | subject_1),
  data = within_matched_raw,
  REML = FALSE
)

# Extract residuals
residuals_raw <- residuals(model_raw)
residuals_log <- residuals(model_log)

# Plot residuals
p_resid_raw <- ggplot(data.frame(fitted = fitted(model_raw), resid = residuals_raw),
                      aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(se = FALSE) +
  labs(title = "Residuals: Raw DTW Distance Model",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal()

p_resid_log <- ggplot(data.frame(fitted = fitted(model_log), resid = residuals_log),
                      aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(se = FALSE) +
  labs(title = "Residuals: Log-Transformed Model",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal()

p_resid <- p_resid_raw | p_resid_log
print(p_resid)
ggsave("analysis_results/diagnostics/residual_plots.png", p_resid, 
       width = 12, height = 5, dpi = 300)

# Q-Q plots for residuals
p_qq_resid_raw <- ggplot(data.frame(resid = residuals_raw), aes(sample = resid)) +
  stat_qq() + stat_qq_line() +
  labs(title = "Residual Q-Q: Raw Model") +
  theme_minimal()

p_qq_resid_log <- ggplot(data.frame(resid = residuals_log), aes(sample = resid)) +
  stat_qq() + stat_qq_line() +
  labs(title = "Residual Q-Q: Log Model") +
  theme_minimal()

p_qq_resid <- p_qq_resid_raw | p_qq_resid_log
print(p_qq_resid)
ggsave("analysis_results/diagnostics/residual_qq_plots.png", p_qq_resid, 
       width = 12, height = 5, dpi = 300)

# 5. Statistical tests for normality
cat("\n=== NORMALITY TESTS ===\n")
cat("Raw DTW Distance:\n")
cat(sprintf("  Shapiro-Wilk (sample of 5000): p = %.4f\n", 
            shapiro.test(sample(dtw_results$dtw_distance, min(5000, nrow(dtw_results))))$p.value))

cat("\nLog-Transformed DTW Distance:\n")
cat(sprintf("  Shapiro-Wilk (sample of 5000): p = %.4f\n", 
            shapiro.test(sample(dtw_results$log_dtw_distance, min(5000, nrow(dtw_results))))$p.value))

cat("\nResiduals from Raw Model:\n")
cat(sprintf("  Shapiro-Wilk: p = %.4f\n", 
            shapiro.test(sample(residuals_raw, min(5000, length(residuals_raw))))$p.value))

cat("\nResiduals from Log Model:\n")
cat(sprintf("  Shapiro-Wilk: p = %.4f\n", 
            shapiro.test(sample(residuals_log, min(5000, length(residuals_log))))$p.value))

# 6. Compare model fit
cat("\n=== MODEL COMPARISON ===\n")
cat("AIC (lower is better):\n")
cat(sprintf("  Raw model: %.2f\n", AIC(model_raw)))
cat(sprintf("  Log model: %.2f\n", AIC(model_log)))

cat("\nBIC (lower is better):\n")
cat(sprintf("  Raw model: %.2f\n", BIC(model_raw)))
cat(sprintf("  Log model: %.2f\n", BIC(model_log)))


# Clear argument for log transformation


# ============================================================================
# QUESTION 1: Within-subject consistency across conditions
# ============================================================================


# Model 1a: Main effects
model_within_main <- lmer(
  log_dtw_distance ~ condition_pair + obstacles_match + (1 | subject_1),
  data = within_data,
  REML = FALSE
)

cat("\n=== WITHIN-SUBJECT CONSISTENCY MODEL (MAIN EFFECTS) ===\n")
print(report(model_within_main, estimator = "ML"))
print(report_performance(model_within_main, estimator = "ML"))
print(report_parameters(model_within_main, estimator = "ML"))

# Model 1b: Focus on matched obstacles
within_matched <- within_data %>%
  filter(obstacles_match == TRUE)

model_within_matched <- lmer(
  log_dtw_distance ~ condition_pair + (1 | subject_1),
  data = within_matched,
  REML = FALSE
)

cat("\n=== WITHIN-SUBJECT CONSISTENCY - MATCHED OBSTACLES ONLY ===\n")
print(report(model_within_matched, estimator = "ML"))
print(report_performance(model_within_matched, estimator = "ML"))
print(report_parameters(model_within_matched, estimator = "ML"))

# Emmeans for matched obstacles (on log scale)
emmeans_within_matched <- emmeans(model_within_matched, specs = "condition_pair")
pairs_within_matched <- pairs(emmeans_within_matched, adjust = "tukey")

cat("\n=== Post-hoc Comparisons (Within-Subject, Matched Obstacles) ===\n")
cat("Note: Estimates are on log scale\n")
print(summary(emmeans_within_matched))
print(summary(pairs_within_matched))

# Back-transform to original scale for interpretation
emmeans_within_matched_original <- emmeans(model_within_matched, specs = "condition_pair", type = "response")
cat("\n=== Estimated Marginal Means (Back-transformed to Original Scale) ===\n")
print(summary(emmeans_within_matched_original))

# Model 1c: Focus on mismatched obstacles (obstacle effect within same condition)
within_mismatched <- within_data %>%
  filter(obstacles_match == FALSE)

model_within_mismatched <- lmer(
  log_dtw_distance ~ condition_pair + (1 | subject_1),
  data = within_mismatched,
  REML = FALSE
)

cat("\n=== WITHIN-SUBJECT CONSISTENCY - OBSTACLE EFFECT ===\n")
print(report(model_within_mismatched, estimator = "ML"))

emmeans_within_mismatched <- emmeans(model_within_mismatched, specs = "condition_pair")
pairs_within_mismatched <- pairs(emmeans_within_mismatched, adjust = "tukey")

cat("\n=== Post-hoc Comparisons (Within-Subject, Obstacle Effect) ===\n")
print(summary(emmeans_within_mismatched))
print(summary(pairs_within_mismatched))

# ============================================================================
# QUESTION 2: Between-subject similarity within conditions
# ============================================================================

between_data <- dtw_results %>%
  filter(dtw_type == "between") %>%
  mutate(
    condition = condition_1,
    obstacles = obstacles_1
  )

model_between <- lmer(
  log_dtw_distance ~ condition * obstacles + (1 | subject_1) + (1 | subject_2),
  data = between_data,
  REML = FALSE
)

cat("\n=== BETWEEN-SUBJECT SIMILARITY MODEL ===\n")
print(report(model_between, estimator = "ML"))
print(report_performance(model_between, estimator = "ML"))
print(report_parameters(model_between, estimator = "ML"))

# Emmeans for between-subject
emmeans_between <- emmeans(model_between, specs = "condition", by = "obstacles")
pairs_between <- pairs(emmeans_between, adjust = "tukey")

cat("\n=== Post-hoc Comparisons (Between-Subject) ===\n")
cat("Note: Estimates are on log scale\n")
print(summary(emmeans_between))
print(summary(pairs_between))

# Back-transform to original scale
emmeans_between_original <- emmeans(model_between, specs = "condition", by = "obstacles", type = "response")
cat("\n=== Estimated Marginal Means (Back-transformed to Original Scale) ===\n")
print(summary(emmeans_between_original))

# Main effect of condition (averaged across obstacles)
emmeans_between_main <- emmeans(model_between, specs = "condition")
pairs_between_main <- pairs(emmeans_between_main, adjust = "tukey")

cat("\n=== Main Effect of Condition (Between-Subject) ===\n")
print(summary(emmeans_between_main))
print(summary(pairs_between_main))

# Back-transform main effect
emmeans_between_main_original <- emmeans(model_between, specs = "condition", type = "response")
cat("\n=== Main Effect (Back-transformed to Original Scale) ===\n")
print(summary(emmeans_between_main_original))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# 1. Within-subject: Matched obstacles (main analysis) - LOG SCALE
p1_log <- ggplot(within_matched, aes(x = condition_pair, y = log_dtw_distance)) +
  geom_violin(aes(fill = condition_pair), alpha = 0.6) +
  geom_boxplot(width = 0.2, outlier.alpha = 0.3) +
  geom_jitter(width = 0.1, alpha = 0.2, size = 1) +
  labs(
    title = "Within-Subject Movement Consistency Across Conditions",
    subtitle = "Matched obstacle conditions only (log scale, lower = more consistent)",
    x = "Condition Pair",
    y = "Log(DTW Distance)",
    fill = "Condition Pair"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

print(p1_log)
ggsave("analysis_results/within_subject_matched_obstacles_log.png", p1_log, width = 8, height = 6, dpi = 300)

# 1b. Within-subject: Matched obstacles - ORIGINAL SCALE
p1_original <- ggplot(within_matched, aes(x = condition_pair, y = dtw_distance)) +
  geom_violin(aes(fill = condition_pair), alpha = 0.6) +
  geom_boxplot(width = 0.2, outlier.alpha = 0.3) +
  geom_jitter(width = 0.1, alpha = 0.2, size = 1) +
  labs(
    title = "Within-Subject Movement Consistency Across Conditions",
    subtitle = "Matched obstacle conditions only (original scale, lower = more consistent)",
    x = "Condition Pair",
    y = "DTW Distance",
    fill = "Condition Pair"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

print(p1_original)
ggsave("analysis_results/within_subject_matched_obstacles_original.png", p1_original, width = 8, height = 6, dpi = 300)

# 2. Within-subject: Obstacle effect (same condition, different obstacles) - LOG SCALE
p2_log <- ggplot(within_mismatched, aes(x = condition_pair, y = log_dtw_distance)) +
  geom_violin(aes(fill = condition_pair), alpha = 0.6) +
  geom_boxplot(width = 0.2, outlier.alpha = 0.3) +
  geom_jitter(width = 0.1, alpha = 0.2, size = 1) +
  labs(
    title = "Within-Subject Movement: Effect of Obstacles",
    subtitle = "Same condition, different obstacles (log scale, lower = obstacles have less impact)",
    x = "Condition (comparing 0 vs 1 obstacles)",
    y = "Log(DTW Distance)",
    fill = "Condition"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

print(p2_log)
ggsave("analysis_results/within_subject_obstacle_effect_log.png", p2_log, width = 8, height = 6, dpi = 300)

# 3. Within-subject: All data comparison - LOG SCALE
p3_log <- ggplot(within_data, aes(x = condition_pair, y = log_dtw_distance, fill = obstacles_match)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  labs(
    title = "Within-Subject Movement Consistency: Full Comparison",
    subtitle = "Log scale: TRUE = same obstacles, FALSE = different obstacles",
    x = "Condition Pair",
    y = "Log(DTW Distance)",
    fill = "Obstacles Match"
  ) +
  scale_fill_manual(values = c("FALSE" = "#E69F00", "TRUE" = "#56B4E9")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p3_log)
ggsave("analysis_results/within_subject_full_comparison_log.png", p3_log, width = 10, height = 6, dpi = 300)

# 4. Between-subject similarity - LOG SCALE
p4_log <- ggplot(between_data, aes(x = condition, y = log_dtw_distance, fill = obstacles)) +
  geom_violin(alpha = 0.6) +
  geom_boxplot(width = 0.3, position = position_dodge(width = 0.9), outlier.alpha = 0.3) +
  labs(
    title = "Between-Subject Movement Similarity Within Conditions",
    subtitle = "Log scale: Lower values = children moving more similarly to each other",
    x = "Condition",
    y = "Log(DTW Distance)",
    fill = "Obstacles"
  ) +
  scale_fill_manual(values = c("0" = "#E69F00", "1" = "#56B4E9")) +
  theme_minimal()

print(p4_log)
ggsave("analysis_results/between_subject_similarity_log.png", p4_log, width = 8, height = 6, dpi = 300)

# 4b. Between-subject similarity - ORIGINAL SCALE
p4_original <- ggplot(between_data, aes(x = condition, y = dtw_distance, fill = obstacles)) +
  geom_violin(alpha = 0.6) +
  geom_boxplot(width = 0.3, position = position_dodge(width = 0.9), outlier.alpha = 0.3) +
  labs(
    title = "Between-Subject Movement Similarity Within Conditions",
    subtitle = "Original scale: Lower values = children moving more similarly to each other",
    x = "Condition",
    y = "DTW Distance",
    fill = "Obstacles"
  ) +
  scale_fill_manual(values = c("0" = "#E69F00", "1" = "#56B4E9")) +
  theme_minimal()

print(p4_original)
ggsave("analysis_results/between_subject_similarity_original.png", p4_original, width = 8, height = 6, dpi = 300)

# 5. Between-subject: Marginal means plot with error bars - LOG SCALE
emmeans_plot_data_log <- as.data.frame(emmeans_between)

p5_log <- ggplot(emmeans_plot_data_log, aes(x = condition, y = emmean, color = obstacles, group = obstacles)) +
  geom_point(size = 3, position = position_dodge(width = 0.3)) +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), 
                width = 0.2, position = position_dodge(width = 0.3)) +
  geom_line(position = position_dodge(width = 0.3)) +
  labs(
    title = "Estimated Marginal Means: Between-Subject Similarity",
    subtitle = "Log scale: Error bars show 95% confidence intervals",
    x = "Condition",
    y = "Estimated Log(DTW Distance)",
    color = "Obstacles"
  ) +
  theme_minimal()

print(p5_log)
ggsave("analysis_results/between_subject_emmeans_log.png", p5_log, width = 8, height = 6, dpi = 300)

# 5b. Between-subject: Marginal means plot - ORIGINAL SCALE (back-transformed)
emmeans_plot_data_original <- as.data.frame(emmeans_between_original)

p5_original <- ggplot(emmeans_plot_data_original, aes(x = condition, y = emmean, color = obstacles, group = obstacles)) +
  geom_point(size = 3, position = position_dodge(width = 0.3)) +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), 
                width = 0.2, position = position_dodge(width = 0.3)) +
  geom_line(position = position_dodge(width = 0.3)) +
  labs(
    title = "Estimated Marginal Means: Between-Subject Similarity",
    subtitle = "Original scale (back-transformed): Error bars show 95% confidence intervals",
    x = "Condition",
    y = "Estimated DTW Distance",
    color = "Obstacles"
  ) +
  theme_minimal()

print(p5_original)
ggsave("analysis_results/between_subject_emmeans_original.png", p5_original, width = 8, height = 6, dpi = 300)

# ============================================================================
# ADDITIONAL DIAGNOSTICS
# ============================================================================

cat("\n=== MODEL DIAGNOSTICS ===\n")

# Check assumptions for within-subject model
p <- plot(check_model(model_within_matched))
# save plot
ggsave("analysis_results/diagnostics/within_subject_model_diagnostics.png", p, width = 10, height = 8, dpi = 300)

# Check assumptions for between-subject model
p <- plot(check_model(model_between))
# save plot
ggsave("analysis_results/diagnostics/between_subject_model_diagnostics.png", p, width = 10, height = 8, dpi = 300)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

cat("\n=== DESCRIPTIVE STATISTICS ===\n")

# Within-subject matched obstacles
within_matched_summary <- within_matched %>%
  group_by(condition_pair) %>%
  summarise(
    n = n(),
    mean_dtw = mean(dtw_distance),
    sd_dtw = sd(dtw_distance),
    median_dtw = median(dtw_distance),
    mean_log_dtw = mean(log_dtw_distance),
    sd_log_dtw = sd(log_dtw_distance),
    median_log_dtw = median(log_dtw_distance),
    .groups = "drop"
  )

print("Within-Subject (Matched Obstacles):")
print(within_matched_summary)

# Between-subject
between_summary <- between_data %>%
  group_by(condition, obstacles) %>%
  summarise(
    n = n(),
    mean_dtw = mean(dtw_distance),
    sd_dtw = sd(dtw_distance),
    median_dtw = median(dtw_distance),
    mean_log_dtw = mean(log_dtw_distance),
    sd_log_dtw = sd(log_dtw_distance),
    median_log_dtw = median(log_dtw_distance),
    .groups = "drop"
  )

print("\nBetween-Subject:")
print(between_summary)

# ============================================================================
# OPTIONAL: Export results to file
# ============================================================================

# Create a results directory
if (!dir.exists("analysis_results")) {
  dir.create("analysis_results")
}

# Save model summaries
sink("analysis_results/model_reports.txt")
cat("=== ANALYSIS NOTE ===\n")
cat("All models use log-transformed DTW distances to handle skewed distributions.\n")
cat("Coefficients are on the log scale. To interpret on the original scale,\n")
cat("exponentiate the coefficients (multiplicative effects on original scale).\n\n")

cat("=== WITHIN-SUBJECT CONSISTENCY - MATCHED OBSTACLES ===\n\n")
print(report(model_within_matched, estimator = "ML"))
cat("\n\n=== WITHIN-SUBJECT POST-HOC COMPARISONS (Log Scale) ===\n\n")
print(summary(pairs_within_matched))
cat("\n\n=== WITHIN-SUBJECT EMMEANS (Back-transformed to Original Scale) ===\n\n")
print(summary(emmeans_within_matched_original))

cat("\n\n=== BETWEEN-SUBJECT SIMILARITY ===\n\n")
print(report(model_between, estimator = "ML"))
cat("\n\n=== BETWEEN-SUBJECT POST-HOC COMPARISONS (Log Scale) ===\n\n")
print(summary(pairs_between_main))
cat("\n\n=== BETWEEN-SUBJECT EMMEANS (Back-transformed to Original Scale) ===\n\n")
print(summary(emmeans_between_main_original))
sink()

# Save summary statistics
write_csv(within_matched_summary, "analysis_results/within_subject_summary.csv")
write_csv(between_summary, "analysis_results/between_subject_summary.csv")

cat("\n✓ Analysis complete! Results saved to analysis_results/\n")
cat("Note: All statistical models used log-transformed DTW distances.\n")
