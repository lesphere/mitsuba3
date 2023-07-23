import drjit as dr
import mitsuba as mi

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def mse(image, image_ref):
    return dr.mean(dr.sqr(image - image_ref))

def main():
    mi.set_variant('cuda_ad_rgb')

    scene = mi.load_file('scenes/cbox.xml', integrator='prb')

    image_ref = mi.render(scene, spp=512)
    mi.Bitmap(image_ref).write('ref.exr')

    params = mi.traverse(scene)

    key = 'green.reflectance.value'

    dr.enable_grad([params[key]])
    params.update()
    image = mi.render(scene, params, spp=128)
    dr.forward(params[key])
    grad_image = dr.grad(image)
    mi.Bitmap(grad_image).write('grad.exr')

    cmap = cm.coolwarm
    vlim = dr.max(dr.abs(grad_image))[0]
    print(f'Remapping colors within range: [{-vlim:.2f}, {vlim:.2f}]')

    fig, axx = plt.subplots(1, 3, figsize=(8, 3))
    for i, ax in enumerate(axx):
        ax.imshow(grad_image[..., i], cmap=cmap, vmin=-vlim, vmax=vlim)
        ax.set_title('RGB'[i] + ' gradients')
        ax.axis('off')
    fig.tight_layout()
    plt.savefig('grad_rgb.png')
    plt.close()

    theta = mi.Float(0.5)
    dr.enable_grad(theta)
    params[key] = mi.Color3f(0.2 * theta, 0.5 * theta, 0.8 * theta)
    params.update()
    dr.forward(theta, dr.ADFlag.ClearEdges)
    image = mi.render(scene, params, spp=128)
    dr.forward_to(image)
    mi.Bitmap(dr.grad(image)).write('grad_theta.exr')

    param_ref = mi.Color3f(params[key])

    params[key] = mi.Color3f(0.01, 0.2, 0.9)
    params.update()

    image_init = mi.render(scene, spp=128)
    mi.Bitmap(image_init).write('init.exr')

    opt = mi.ad.Adam(lr=0.05)
    opt[key] = params[key]
    params.update(opt)

    iters = 50

    losses = []

    for it in range(iters):
        image = mi.render(scene, params, spp=4)
        loss = mse(image, image_ref)
        dr.backward(loss)
        opt.step()
        opt[key] = dr.clamp(opt[key], 0, 1)
        params.update(opt)

        loss = dr.sum(dr.sqr(param_ref - params[key]))
        print(f'Iteration {it:02d}: parameter error = {loss[0]:.6f}', end='\r')
        losses.append(loss)

    image_final = mi.render(scene, spp=128)
    mi.Bitmap(image_final).write('final.exr')

    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('MSE(param)')
    plt.title('Parameter loss plot')
    plt.savefig('loss.png')
    plt.cla()

    print('\nOptimization complete.')

if __name__ == '__main__':
    main()
