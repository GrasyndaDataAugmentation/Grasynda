library(ggraph)
library(tidygraph)
library(tidyverse)
library(igraph)
path = ""
df <- read.csv(path)
df$from <- paste0('Q', df$from+1)
df$to <- paste0('Q', df$to+1)

# Check if weight column exists, if not create it
if(!"weight" %in% colnames(df)) {
  # Try common alternative names
  if("value" %in% colnames(df)) {
    df$weight <- df$value
  } else if("count" %in% colnames(df)) {
    df$weight <- df$count
  } else if("probability" %in% colnames(df)) {
    df$weight <- df$probability
  } else if("freq" %in% colnames(df)) {
    df$weight <- df$freq
  } else {
    # If no weight column found, set all to 1
    df$weight <- 1
  }
}

# Create directed graph
graph <- as_tbl_graph(df, directed = TRUE) |>
  activate(nodes) |>
  # Calculate popularity (in-degree)
  mutate(Popularity = centrality_degree(mode = 'in'))

# Plot the graph
ggraph(graph, layout = 'nicely') +
  # Add edges with arrows - now using weight for edge width
  geom_edge_fan(
    aes(width = weight, alpha = 0.7),  # Use weight for edge width
    arrow = arrow(length = unit(4, 'mm'), type = "closed"),
    end_cap = circle(3, 'mm'),
    start_cap = circle(3, 'mm'),
    show.legend = FALSE,
    edge_colour = "grey20"  # Darker edge color for better visibility
  ) +
  # Scale edge width
  scale_edge_width(range = c(0.2, 2)) +  # Adjust edge width scaling

  # Enhance nodes with fixed size based on popularity
  geom_node_point(
    aes(size = Popularity),
    color = "white",
    fill = "steelblue",
    shape = 21,
    stroke = 0.8  # Slightly thicker stroke for better definition
  ) +

  # Add labels inside nodes
  geom_node_text(
    aes(label = name),
    color = "white",
    size = 3.5,  # Slightly larger text
    repel = FALSE,
    fontface = "bold"
  ) +

  # Scale node sizes - fixed minimum size to ensure all labels fit
  scale_size_continuous(
    range = c(8, 14),  # Increased minimum size to fit all labels
    breaks = seq(0, max(graph %>% activate(nodes) %>% pull(Popularity)), by = 2)
  ) +

  # Clean minimal theme with white background
  theme_graph(
    base_family = "serif",
    background = 'white',
    text_colour = "black"
  ) +

  # Remove legends
  guides(size = "none", edge_width = "none", edge_alpha = "none") +

  # Ensure clean white background
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  )
