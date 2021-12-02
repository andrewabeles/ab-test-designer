library(shiny)
library(ggplot2)
library(plotly)

ui <- navbarPage("Sample Size Estimator",
        tabPanel("Proportions",
            sidebarLayout(
                sidebarPanel(
                    numericInput("users_per_week_proportion",
                                 "Users per Week",
                                 value = 100000),
                    sliderInput("traffic_pct_proportion",
                                "Percent of Traffic",
                                min = 0,
                                max = 1,
                                value = 1),
                    sliderInput("baseline_proportion",
                                "Baseline",
                                min = 0,
                                max = 1,
                                value = 0.5),
                    sliderInput("alpha_proportion",
                                "Alpha",
                                min = 0, 
                                max = 1,
                                value = 0.05),
                    sliderInput("power_proportion",
                                "Power",
                                min = 0, 
                                max = 1,
                                value = 0.8)
                ),
                mainPanel(
                    plotlyOutput("plot_proportions"),
                    sliderInput("effect_size_range_proportion",
                                "Effect Size Range",
                                width = "100%",
                                min = 0.01,
                                max = 1,
                                value = c(0.05, 0.25))
                )
            )
        ),
        tabPanel("Means",
            sidebarLayout(
                sidebarPanel(
                    numericInput("users_per_week_mean",
                                 "Users per Week",
                                 value = 100000),
                    sliderInput("traffic_pct_mean",
                                "Percent of Traffic",
                                min = 0,
                                max = 1,
                                value = 1),
                    numericInput("baseline_mean",
                                 "Baseline",
                                 value = 3.14),
                    numericInput("sd",
                                 "Standard Deviation",
                                 value = 2.71),
                    sliderInput("alpha_mean",
                                "Alpha",
                                min = 0, 
                                max = 1,
                                value = 0.05),
                    sliderInput("power_mean",
                                "Power",
                                min = 0, 
                                max = 1,
                                value = 0.8)
                ),
                mainPanel(
                    plotlyOutput("plot_means"),
                    sliderInput("effect_size_range_mean",
                                "Effect Size Range",
                                width = "100%",
                                min = 0.01,
                                max = 1,
                                value = c(0.05, 0.25))
                )
            )
        ),
        tabPanel("About",
                 p("This is an app for estimating the sample size needed to run an experiment. 
                   It takes the following parameters as input: baseline, standard deviation (only for means), alpha, and power.
                   It outputs the sample size per group required to detect a given effect size."),
                 h3("Baseline"),
                 p("The baseline value of the variable being tested. The variable's proportion or average in the control group."),
                 h3("Standard Deviation"),
                 p("The standard deviation of the baseline variable. 
                   Larger standard deviations require larger sample sizes to ensure the observed difference between groups is not due to the baseline variable's natural variance."),
                 h3("Alpha"),
                 p("Also known as the significance level, alpha is the experiment's Type I error (false positive) rate."),
                 h3("Power"),
                 p("1 - Beta, the Type II error or false negative rate. Power represents the probability the experiment will detect a given effect size at a given significance level.")
        )
    )

server <- function(input, output) {
    output$plot_proportions <- renderPlotly({
        p1 <- input$baseline_proportion
        effect_size_range <- input$effect_size_range_proportion
        effect_sizes <- seq(effect_size_range[1], effect_size_range[2], by = 0.01)
        sample_sizes <- c()
        for (e in effect_sizes) {
            p2 <- p1 * (1 + e)
            if (p2 > 1) {p2 <- 1}
            n <- power.prop.test(
                n = NULL,
                p1 = p1,
                p2 = p2,
                sig.level = input$alpha_proportion,
                power = input$power_proportion
            )$n * 2 # the function outputs required sample size per group, so we double to get the total required sample size 
            sample_sizes <- append(sample_sizes, n)
        }
        users_per_week <- input$users_per_week_proportion * input$traffic_pct_proportion 
        weeks <- sample_sizes / users_per_week
        data <- data.frame(effect_sizes, sample_sizes, weeks)
        colnames(data) <- c("EffectSize", "SampleSize", "Weeks")
        ggplot(data, aes(x = EffectSize, y = Weeks, text = paste("SampleSize:", SampleSize))) + 
            geom_line(color = "grey") + 
            geom_point(color = "coral") +
            xlab("Effect Size") + 
            ylab("Weeks")
    })
    output$plot_means <- renderPlotly({
        x1 <- input$baseline_mean
        effect_size_range <- input$effect_size_range_mean
        effect_sizes <- seq(effect_size_range[1], effect_size_range[2], by = 0.01)
        sample_sizes <- c()
        for (e in effect_sizes) {
            x2 <- x1 * (1 + e)
            n <- power.t.test(
                n = NULL, 
                delta = x2 - x1,
                sd = input$sd,
                sig.level = input$alpha_mean,
                power = input$power_mean
            )$n * 2 # the function outputs required sample size per group, so we double to get the total required sample size 
            sample_sizes <- append(sample_sizes, n)
        }
        users_per_week <- input$users_per_week_mean * input$traffic_pct_mean 
        weeks <- sample_sizes / users_per_week
        data <- data.frame(effect_sizes, sample_sizes, weeks)
        colnames(data) <- c("EffectSize", "SampleSize", "Weeks")
        ggplot(data, aes(x = EffectSize, y = Weeks, text = paste("SampleSize:", SampleSize))) +
            geom_line(color = "grey") +
            geom_point(color = "coral") + 
            xlab("Effect Size") + 
            ylab("Weeks")
    })
}

shinyApp(ui = ui, server = server)
