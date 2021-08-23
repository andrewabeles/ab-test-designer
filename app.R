library(shiny)
library(ggplot2)
library(plotly)

ui <- navbarPage("Sample Size Estimator",
        tabPanel("Proportions",
            sidebarLayout(
                sidebarPanel(
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
                   plotlyOutput("plot_proportions")
                )
            )
        ),
        tabPanel("Means",
                 sidebarLayout(
                     sidebarPanel(
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
                         plotlyOutput("plot_means")
                     )
                 )
        ),
        tabPanel("About",
                 p("This is an app for estimating the sample size needed to run an experiment. 
                   It takes 3 parameters as input: baseline, alpha, and power.
                   It outputs the sample size per group required to detect a given effect size."),
                 h3("Baseline"),
                 p("The baseline value of the variable being tested. The variable's proportion or average in the control group."),
                 h3("Alpha"),
                 p("Also known as the significance level, alpha is the experiment's Type I error (false positive) rate."),
                 h3("Power"),
                 p("1 - Beta, the Type II error or false negative rate. Power represents the probability the experiment will detect a given effect size at a given significance level.")
        )
    )

server <- function(input, output) {
    output$plot_proportions <- renderPlotly({
        p1 <- input$baseline_proportion
        effect_sizes <- seq(0.01, 1, by = 0.01)
        sample_sizes <- c()
        for (e in effect_sizes) {
            p2 <- p1 * (1 + e)
            n <- power.prop.test(
                n = NULL,
                p1 = p1,
                p2 = p2,
                sig.level = input$alpha_proportion,
                power = input$power_proportion
            )$n
            sample_sizes <- append(sample_sizes, n)
        }
        data <- data.frame(effect_sizes, sample_sizes)
        ggplot(data, aes(x = effect_sizes, y = sample_sizes)) + 
            geom_line(color = "grey") + 
            geom_point(color = "coral") +
            xlab("Effect Size") + 
            ylab("Sample Size")
    })
    output$plot_means <- renderPlotly({
        x1 <- input$baseline_mean
        effect_sizes <- seq(0.01, 1, by = 0.01)
        sample_sizes <- c()
        for (e in effect_sizes) {
            x2 <- x1 * (1 + e)
            n <- power.t.test(
                n = NULL, 
                delta = x2 - x1,
                sd = input$sd,
                sig.level = input$alpha_mean,
                power = input$power_mean
            )$n
            sample_sizes <- append(sample_sizes, n)
        }
        data <- data.frame(effect_sizes, sample_sizes)
        ggplot(data, aes(x = effect_sizes, y = sample_sizes)) +
            geom_line(color = "grey") +
            geom_point(color = "coral") + 
            xlab("Effect Size") + 
            ylab("Sample Size")
    })
}

shinyApp(ui = ui, server = server)
