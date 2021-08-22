library(shiny)
library(ggplot2)
library(plotly)

ui <- navbarPage("Sample Size Estimator",
        tabPanel("Proportions",
            sidebarLayout(
                sidebarPanel(
                    sliderInput("baseline",
                                "Baseline",
                                min = 0,
                                max = 1,
                                value = 0.5),
                    sliderInput("alpha",
                                "Alpha",
                                min = 0, 
                                max = 1,
                                value = 0.05),
                    sliderInput("power",
                                "Power",
                                min = 0, 
                                max = 1,
                                value = 0.8)
                ),
                mainPanel(
                   plotlyOutput("plot")
                )
            )
        ),
        tabPanel("About",
                 p("This is an app for estimating the sample size needed to run an experiment. 
                   It takes 3 parameters as input: baseline, alpha, and power."),
                 h3("Baseline"),
                 p("The baseline value of the variable being tested. The variable's proportion or average in the control group."),
                 h3("Alpha"),
                 p("Also known as the significance level, alpha is the experiment's Type I error (false positive) rate."),
                 h3("Power"),
                 p("1 - Beta, the Type II error or false negative rate. Power represents the probability the experiment will detect a given effect size at a given significance level.")
        )
    )

server <- function(input, output) {
    output$plot <- renderPlotly({
        effect_sizes <- seq(0.01, 1, by = 0.01)
        sample_sizes <- c()
        for (e in effect_sizes) {
            p2 <- input$baseline * (1 + e) 
            if (p2 > 1) {
                p2 <- 1
            }
            n <- power.prop.test(
                n = NULL,
                p1 = input$baseline,
                p2 = p2,
                sig.level = input$alpha,
                power = input$power
            )$n
            sample_sizes <- append(sample_sizes, n)
        }
        data <- data.frame(effect_sizes, sample_sizes)
        ggplot(data, aes(x = effect_sizes, y = sample_sizes)) + 
            geom_line(color="grey") + 
            geom_point(color="coral") +
            xlab("Effect Size") + 
            ylab("Sample Size")
    })
}

shinyApp(ui = ui, server = server)
