import numpy as np

# image urls
image_urls = [
    "https://drive.usercontent.google.com/download?id=1TICo_0Cp-CDh22OBFS4s-LzUecKrxNJi&authuser=0",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRRW76Uoole_PZ9xzWCzmKYnTa_YUyJTGOmHg&s",
    "https://i.pinimg.com/474x/1d/27/a3/1d27a361abdbe6161f867fe81e3c273e.jpg",
    "https://i.pinimg.com/474x/45/f3/de/45f3dede26bc06804162a5e476d45db2.jpg",
    "https://i.pinimg.com/474x/1a/74/95/1a74955ee768dcb930c2be180c4dd3b2.jpg",
    "https://i.pinimg.com/236x/d6/cf/4e/d6cf4ed0a5642d4afd58a7c41bcad583.jpg",
    "https://i.pinimg.com/474x/c2/56/cc/c256cc070d807f343ce7f0af9f273c1e.jpg",
    "https://i.pinimg.com/236x/e5/29/a8/e529a89194956fb83727d8ac5c69264c.jpg",
    "https://images.unsplash.com/photo-1731082417879-710ff0c868ae?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxmZWF0dXJlZC1waG90b3MtZmVlZHw0M3x8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1731485003527-851dd35bcf29?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxmZWF0dXJlZC1waG90b3MtZmVlZHw4fHx8ZW58MHx8fHx8",
    "https://images.unsplash.com/photo-1731331203151-a7e914e02d23?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxmZWF0dXJlZC1waG90b3MtZmVlZHw0fHx8ZW58MHx8fHx8",
    "https://cdn.stocksnap.io/img-thumbs/280h/team-meeting_VQXYE2ZEHC.jpg",
    "https://cdn.stocksnap.io/img-thumbs/280h/pizza-wine_IJESKJTYB6.jpg",
    "https://kaboompics.com/cache/1/8/4/3/0/18430481f85efedeff8c1487c84c00d6e7943ffe.jpeg",
    "https://cdn.stocksnap.io/img-thumbs/280h/healthy-smoothie_49CRAQPKUQ.jpg",
    "https://cdn.stocksnap.io/img-thumbs/280h/8Y0EDX4VP9.jpg",
    "https://kaboompics.com/cache/1/9/a/8/f/19a8fa465151a5ae753ad26793da09fb270ee620.jpeg",
    "https://cdn.stocksnap.io/img-thumbs/280h/HHZ5NPNR1T.jpg",
    "https://cdn.stocksnap.io/img-thumbs/280h/3ZHG0XOIT6.jpg",
    "https://kaboompics.com/cache/8/e/b/8/9/8eb89015020587d21f7aceb6a48371a029cbbf7a.jpeg",
    "https://cdn.stocksnap.io/img-thumbs/280h/SCC00WCQ3I.jpg",
    "https://cdn.stocksnap.io/img-thumbs/280h/RAW1RLRTM7.jpg",
    "https://cdn.stocksnap.io/img-thumbs/280h/KPYGENZRS6.jpg",
    "https://cdn.stocksnap.io/img-thumbs/280h/ITA18FXIBL.jpg",
    "https://cdn.stocksnap.io/img-thumbs/280h/BHIL9FV6RK.jpg",
]

# cosine similarity function
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

