library(magick)

crop_all_versions <- function(
    dir_in = ".",
    width, height,
    x, y,
    anchor = c("topleft", "center"),
    dx = 0, dy = 0,
    pattern = "\\.(jpg|jpeg|tif|tiff|png)$",
    preview = TRUE,
    preview_file = NULL,
    overwrite = FALSE,
    prefixes_to_remove = c(
      "2025_01_21_09-29-49_cropped_enhanced_?",
      "cropped_enhance_?",
      "hand_edited_?"
    ),
    name_len_threshold = 16,    # keep names < threshold intact
    pad_x = 10, pad_y = 10      # padding for the text label
) {
  anchor <- match.arg(anchor)
  
  files <- list.files(dir_in, pattern = pattern, ignore.case = TRUE, full.names = TRUE)
  if (!length(files)) stop("No image files found in: ", dir_in)
  
  # Output folder: patches/
  dir_out <- file.path(dir_in, "patches")
  if (!dir.exists(dir_out)) dir.create(dir_out, recursive = TRUE)
  
  # Build magick geometry string
  to_geometry <- function(img_path) {
    if (anchor == "center") {
      x0 <- round(x - width/2)
      y0 <- round(y - height/2)
    } else {
      x0 <- x
      y0 <- y
    }
    sprintf("%dx%d+%d+%d", as.integer(width), as.integer(height), round(x0 + dx), round(y0 + dy))
  }
  
  # Filename cleanup per your rules
  clean_base_name <- function(name_no_ext, prefixes) {
    if (nchar(name_no_ext) >= name_len_threshold) {
      out <- name_no_ext
      for (p in prefixes) out <- sub(paste0("^", p), "", out, perl = TRUE)
      out <- sub("^_+", "", out)
      if (nchar(out) == 0) out <- name_no_ext
      return(out)
    } else {
      return(name_no_ext)
    }
  }
  
  # --- Preview on one file (visual check) ---
  if (isTRUE(preview)) {
    fprev <- if (!is.null(preview_file)) preview_file else files[1]
    geom  <- to_geometry(fprev)
    message("Preview geometry: ", geom)
    
    im    <- image_read(fprev)
    crop  <- image_crop(im, geom)
    
    # Red box overlay on original
    parts <- as.integer(unlist(strsplit(gsub("[x+]", " ", geom), " ")))
    w <- parts[1]; h <- parts[2]; xg <- parts[3]; yg <- parts[4]
    overlay <- image_draw(im); rect(xg, yg, xg+w, yg+h, border="red", lwd=6); dev.off()
    
    # Label (clean name, no extension)
    base <- basename(fprev)
    name_no_ext <- tools::file_path_sans_ext(base)
    label <- clean_base_name(name_no_ext, prefixes_to_remove)
    
    info <- image_info(crop)
    label_size <- max(10, round(info$height / 12))
    crop_annot <- image_annotate(
      crop,
      text = label,
      size = label_size,
      color = "white",
      gravity = "northwest",
      location = sprintf("+%d+%d", pad_x, pad_y),
      strokecolor = "black",
      weight = 700
    )
    
    print(overlay)
    print(crop_annot)
    message("If the preview looks good, run again with preview = FALSE to process all files.")
    return(invisible(list(geometry = geom, preview_file = fprev, out_dir = dir_out)))
  }
  
  # --- Batch process to PNG ---
  for (f in files) {
    geom <- to_geometry(f)
    img  <- image_read(f)
    cropped <- image_crop(img, geometry = geom)
    
    # Determine output name (PNG) and label
    base <- basename(f)
    name_no_ext <- tools::file_path_sans_ext(base)
    clean_name <- clean_base_name(name_no_ext, prefixes_to_remove)
    out_path   <- file.path(dir_out, paste0(clean_name, ".png"))
    
    if (!overwrite && file.exists(out_path)) {
      message("Skipping existing file: ", out_path)
      next
    }
    
    # Annotate (label is clean base name only)
    info <- image_info(cropped)
    label_size <- max(10, round(info$height / 12))
    cropped_annot <- image_annotate(
      cropped,
      text = clean_name,
      size = label_size,
      color = "white",
      gravity = "northwest",
      location = sprintf("+%d+%d", pad_x, pad_y),
      strokecolor = "black",
      weight = 700
    )
    
    # Preserve high bit depth where possible (16-bit PNG if source >= ~12-bit)
    depth_out <- if (!is.null(info$depth) && is.finite(info$depth) && info$depth >= 12) 16 else 8
    
    image_write(cropped_annot, path = out_path, format = "png", depth = depth_out)
    message(sprintf("Wrote (PNG, %d-bit): %s", depth_out, out_path))
  }
  
  invisible(dir_out)
}




crop_all_versions(
  dir_in = ".",
  width = 1400, height = 800,
  x = 800, y = 2000,          # with anchor="topleft", this is the patch's top-left corner
  anchor = "topleft",
  dx = 0, dy = 0,            # nudge if needed (e.g., dx=+20, dy=+10 moves right & down)
  preview = FALSE             # shows red box and an annotated sample crop; does NOT batch-write
)